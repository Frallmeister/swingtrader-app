"""Construction of aligned, unsplit temporal modeling datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.engine import Engine

from swingtrader.data.features.contracts import FeatureSetSpec
from swingtrader.data.market_frame import (
    MARKET_PRICE_INDEX_NAMES,
    validate_market_price_index,
)
from swingtrader.modeling.datasets.labels import generate_target_set
from swingtrader.modeling.datasets.specifications import TemporalDatasetSpec

TARGET_END_DATE_COLUMN = "target_end_date"
TRAINING_ELIGIBLE_COLUMN = "training_eligible_at_cutoff"
TRAINING_ELIGIBILITY_REASONS_COLUMN = "training_eligibility_reasons"
SAMPLE_METADATA_COLUMNS = (
    TARGET_END_DATE_COLUMN,
    TRAINING_ELIGIBLE_COLUMN,
    TRAINING_ELIGIBILITY_REASONS_COLUMN,
)


@dataclass(frozen=True, slots=True)
class TickerEligibility:
    """Cutoff-aware training eligibility metadata for one ticker."""

    ticker: str
    eligible: bool
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker.strip():
            raise ValueError("Eligibility ticker must be a non-empty string.")
        object.__setattr__(self, "ticker", self.ticker.strip())
        if not isinstance(self.eligible, bool):
            raise TypeError("Eligibility status must be a boolean.")
        if isinstance(self.failure_reasons, str):
            raise TypeError("Eligibility failure reasons must be an iterable of strings.")
        try:
            reasons = tuple(self.failure_reasons)
        except TypeError as exc:
            raise TypeError("Eligibility failure reasons must be an iterable of strings.") from exc
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("Eligibility failure reasons must be non-empty strings.")
        if len(reasons) != len(set(reasons)):
            raise ValueError("Eligibility failure reasons must be unique.")
        if self.eligible and reasons:
            raise ValueError("Eligible tickers must not declare failure reasons.")
        if not self.eligible and not reasons:
            raise ValueError("Ineligible tickers must declare at least one failure reason.")
        object.__setattr__(self, "failure_reasons", tuple(sorted(reasons)))


@dataclass(frozen=True, slots=True)
class TemporalDatasetManifest:
    """Deterministic construction summary for one temporal dataset bundle."""

    spec: TemporalDatasetSpec
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    sample_columns: tuple[str, ...]
    source_row_count: int
    sample_row_count: int
    excluded_missing_target_count: int
    observed_ticker_count: int
    signal_date_start: date | None
    signal_date_end: date | None
    target_end_date_start: date | None
    target_end_date_end: date | None
    feature_missing_counts: tuple[tuple[str, int], ...]
    selected_target_summary: tuple[tuple[str, int | float], ...]
    eligible_ticker_count: int
    eligibility_failure_counts: tuple[tuple[str, int], ...]

    def to_manifest(self) -> dict[str, object]:
        """Return a JSON-serializable bundle manifest."""
        return {
            "manifest_schema_version": 1,
            "dataset_spec": {**self.spec.to_manifest(), "digest": self.spec.digest},
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "sample_columns": list(self.sample_columns),
            "source_row_count": self.source_row_count,
            "sample_row_count": self.sample_row_count,
            "excluded_missing_target_count": self.excluded_missing_target_count,
            "observed_ticker_count": self.observed_ticker_count,
            "signal_date_range": _date_range(self.signal_date_start, self.signal_date_end),
            "target_end_date_range": _date_range(
                self.target_end_date_start,
                self.target_end_date_end,
            ),
            "feature_missing_counts": dict(self.feature_missing_counts),
            "selected_target_summary": dict(self.selected_target_summary),
            "eligible_ticker_count": self.eligible_ticker_count,
            "eligibility_failure_counts": dict(self.eligibility_failure_counts),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the bundle manifest with deterministic key ordering."""
        return json.dumps(self.to_manifest(), indent=indent, sort_keys=True)

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical bundle manifest."""
        payload = json.dumps(self.to_manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TemporalDatasetBundle:
    """Own aligned feature, target, and sample-metadata frames.

    All frames use the same canonical sample index. The selected supervised
    target is complete, while feature warm-up and source-quality missing values
    are intentionally retained for later split-aware preprocessing.
    """

    features: pd.DataFrame
    targets: pd.DataFrame
    samples: pd.DataFrame
    manifest: TemporalDatasetManifest

    def __post_init__(self) -> None:
        features = self.features.copy(deep=True)
        targets = self.targets.copy(deep=True)
        samples = self.samples.copy(deep=True)
        _validate_bundle_frames(features, targets, samples, self.manifest)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "samples", samples)


def build_temporal_dataset(
    *,
    engine: Engine,
    spec: TemporalDatasetSpec,
) -> TemporalDatasetBundle:
    """Build an unsplit temporal dataset from bronze market data.

    The complete available history through ``spec.data_cutoff`` is loaded for
    the exact resolved universe. Features, targets, and training eligibility
    are therefore evaluated against the same inclusive data cutoff.

    Parameters
    ----------
    engine
        SQLAlchemy engine containing the bronze daily-price table.
    spec
        Immutable description of the feature set, target set, supervised
        task, resolved universe, and data cutoff.

    Returns
    -------
    TemporalDatasetBundle
        Aligned feature, target, and sample-metadata frames together with the
        deterministic dataset manifest.

    Raises
    ------
    ValueError
        If a declared ticker is missing or the loaded data violates a source,
        target, or temporal contract.
    """
    from swingtrader.data.bronze.loaders import load_bronze_daily_prices
    from swingtrader.data.eligibility import (
        TrainingEligibilityStatus,
        check_training_eligibility,
    )

    source_columns = _source_value_columns(spec)
    loaded = load_bronze_daily_prices(
        engine=engine,
        provider=spec.universe.provider,
        tickers=spec.universe.tickers,
        end_date=spec.data_cutoff,
        columns=source_columns,
    )
    prices = loaded.set_index(list(MARKET_PRICE_INDEX_NAMES)).sort_index()
    eligibility_result = check_training_eligibility(
        provider=spec.universe.provider,
        tickers=spec.universe.tickers,
        engine=engine,
        data_cutoff=spec.data_cutoff,
    )
    eligibility = {
        state.ticker: TickerEligibility(
            ticker=state.ticker,
            eligible=state.status == TrainingEligibilityStatus.ELIGIBLE,
            failure_reasons=tuple(reason.value for reason in state.failure_reasons),
        )
        for state in eligibility_result.states
    }
    return construct_temporal_dataset(prices, spec=spec, eligibility=eligibility)


def construct_temporal_dataset(
    prices: pd.DataFrame,
    *,
    spec: TemporalDatasetSpec,
    eligibility: Mapping[str, TickerEligibility],
) -> TemporalDatasetBundle:
    """Construct an unsplit temporal dataset from an in-memory market frame.

    The input must use the canonical market index schema, although its rows
    may be unordered. The constructor sorts the frame, calculates features
    and targets independently over the full history, and retains only rows
    where the selected supervised target and its resolution date exist.
    Feature missing values are preserved.

    Parameters
    ----------
    prices
        Market history indexed by provider, ticker, and trading date.
    spec
        Immutable temporal dataset specification.
    eligibility
        Cutoff-aware eligibility metadata for every ticker in the resolved
        universe.

    Returns
    -------
    TemporalDatasetBundle
        Independently owned and index-aligned dataset frames and manifest.

    Raises
    ------
    ValueError
        If the source frame, eligibility mapping, generated columns, aligned
        indexes, or target-resolution dates violate the dataset contract.
    """
    prices = prices.copy(deep=True).sort_index()
    validate_market_price_index(prices)
    _validate_source_scope(prices, spec=spec, eligibility=eligibility)

    feature_result = _generate_feature_set(
        prices.copy(deep=True),
        feature_set=spec.feature_set,
    )
    target_result = generate_target_set(
        prices.copy(deep=True),
        target_set=spec.target_set,
    )
    _validate_unchanged_index(prices.index, target_result, family="Target set")

    features = feature_result.loc[:, spec.feature_set.feature_columns].copy()
    targets = target_result.loc[:, spec.target_set.target_columns].copy()
    target_end_dates = _target_end_dates(targets, spec=spec)

    selected_target = targets[spec.task.target_column]
    retained = selected_target.notna()
    if not retained.any():
        raise ValueError("No rows have an available selected supervised target.")
    if target_end_dates.loc[retained].isna().any():
        raise ValueError("Selected targets must have a corresponding target end date.")

    samples = _sample_metadata(
        prices.index,
        target_end_dates=target_end_dates,
        eligibility=eligibility,
    )
    source_row_count = len(prices)
    features = features.loc[retained].copy()
    targets = targets.loc[retained].copy()
    samples = samples.loc[retained].copy()

    _validate_target_dates(samples, data_cutoff=spec.data_cutoff)
    manifest = _build_manifest(
        spec=spec,
        features=features,
        targets=targets,
        samples=samples,
        source_row_count=source_row_count,
        eligibility=eligibility,
    )
    return TemporalDatasetBundle(
        features=features,
        targets=targets,
        samples=samples,
        manifest=manifest,
    )


def _source_value_columns(spec: TemporalDatasetSpec) -> tuple[str, ...]:
    names = set(spec.feature_set.source_columns).union(spec.target_set.source_columns)
    names.difference_update(MARKET_PRICE_INDEX_NAMES)
    return tuple(sorted(names))


def _generate_feature_set(
    prices: pd.DataFrame,
    *,
    feature_set: FeatureSetSpec,
) -> pd.DataFrame:
    result = prices
    for block in feature_set.blocks:
        available = set(result.columns).union(MARKET_PRICE_INDEX_NAMES)
        missing = sorted(block.required_columns.difference(available))
        if missing:
            raise ValueError(
                f"Feature block {block.name!r} is missing required inputs: {', '.join(missing)}"
            )
        collisions = sorted(set(block.output_columns).intersection(result.columns))
        if collisions:
            raise ValueError(
                f"Feature block {block.name!r} would overwrite columns: {', '.join(collisions)}"
            )
        previous_index = result.index.copy()
        result = block.apply(result)
        _validate_unchanged_index(
            previous_index,
            result,
            family=f"Feature block {block.name!r}",
        )
        missing_outputs = sorted(set(block.output_columns).difference(result.columns))
        if missing_outputs:
            raise ValueError(
                f"Feature block {block.name!r} did not produce columns: "
                f"{', '.join(missing_outputs)}"
            )
    return result


def _validate_source_scope(
    prices: pd.DataFrame,
    *,
    spec: TemporalDatasetSpec,
    eligibility: Mapping[str, TickerEligibility],
) -> None:
    if prices.empty:
        raise ValueError("Temporal dataset source history must not be empty.")
    providers = set(prices.index.get_level_values("provider"))
    if providers != {spec.universe.provider}:
        raise ValueError("Source history must contain exactly the universe provider.")
    observed_tickers = set(prices.index.get_level_values("ticker"))
    missing_tickers = sorted(set(spec.universe.tickers).difference(observed_tickers))
    unexpected_tickers = sorted(observed_tickers.difference(spec.universe.tickers))
    if missing_tickers:
        names = ", ".join(missing_tickers)
        raise ValueError(f"Source history is missing universe tickers: {names}")
    if unexpected_tickers:
        names = ", ".join(unexpected_tickers)
        raise ValueError(f"Source history contains unexpected tickers: {names}")
    signal_dates = pd.DatetimeIndex(prices.index.get_level_values("trading_date"))
    if signal_dates.max().date() > spec.data_cutoff:
        raise ValueError("Source history contains rows after the dataset cutoff.")
    eligibility_tickers = set(eligibility)
    if eligibility_tickers != set(spec.universe.tickers):
        raise ValueError("Eligibility metadata must cover exactly the universe tickers.")
    for ticker, state in eligibility.items():
        if ticker != state.ticker:
            raise ValueError("Eligibility mapping keys must match their ticker states.")


def _validate_unchanged_index(
    before: pd.Index,
    after: pd.DataFrame,
    *,
    family: str,
) -> None:
    if not before.equals(after.index):
        raise ValueError(f"{family} changed the canonical sample index.")
    validate_market_price_index(after)


def _target_end_dates(
    targets: pd.DataFrame,
    *,
    spec: TemporalDatasetSpec,
) -> pd.Series:
    column = spec.task.target_end_date_column
    if column is not None:
        values = pd.to_datetime(targets[column], errors="coerce")
        return pd.Series(values, index=targets.index, name=TARGET_END_DATE_COLUMN)

    horizon = spec.task.horizon_sessions
    if horizon is None:  # protected by TemporalDatasetSpec validation
        raise ValueError("Temporal dataset tasks must declare horizon_sessions.")
    dates = pd.Series(
        pd.DatetimeIndex(targets.index.get_level_values("trading_date")),
        index=targets.index,
        name=TARGET_END_DATE_COLUMN,
    )
    return dates.groupby(level=["provider", "ticker"], sort=False).shift(-horizon)


def _sample_metadata(
    index: pd.MultiIndex,
    *,
    target_end_dates: pd.Series,
    eligibility: Mapping[str, TickerEligibility],
) -> pd.DataFrame:
    tickers = index.get_level_values("ticker")
    eligible = pd.Series(
        [eligibility[str(ticker)].eligible for ticker in tickers],
        index=index,
        dtype="boolean",
    )
    reasons = pd.Series(
        [eligibility[str(ticker)].failure_reasons for ticker in tickers],
        index=index,
        dtype="object",
    )
    return pd.DataFrame(
        {
            TARGET_END_DATE_COLUMN: pd.to_datetime(target_end_dates),
            TRAINING_ELIGIBLE_COLUMN: eligible,
            TRAINING_ELIGIBILITY_REASONS_COLUMN: reasons,
        },
        index=index,
    )


def _validate_target_dates(samples: pd.DataFrame, *, data_cutoff: date) -> None:
    signal_dates = pd.DatetimeIndex(samples.index.get_level_values("trading_date"))
    target_end_dates = pd.DatetimeIndex(samples[TARGET_END_DATE_COLUMN])
    if target_end_dates.hasnans:
        raise ValueError("Retained samples must have non-missing target end dates.")
    if (target_end_dates <= signal_dates).any():
        raise ValueError("Target end dates must follow their signal dates.")
    if target_end_dates.max().date() > data_cutoff:
        raise ValueError("Target end dates must not exceed the dataset cutoff.")


def _build_manifest(
    *,
    spec: TemporalDatasetSpec,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    samples: pd.DataFrame,
    source_row_count: int,
    eligibility: Mapping[str, TickerEligibility],
) -> TemporalDatasetManifest:
    signal_dates = pd.DatetimeIndex(samples.index.get_level_values("trading_date"))
    target_end_dates = pd.DatetimeIndex(samples[TARGET_END_DATE_COLUMN])
    selected_summary = _selected_target_summary(
        targets[spec.task.target_column],
        task_type=spec.task.task_type,
    )
    failure_counts = Counter(
        reason for state in eligibility.values() for reason in state.failure_reasons
    )
    return TemporalDatasetManifest(
        spec=spec,
        feature_columns=tuple(features.columns),
        target_columns=tuple(targets.columns),
        sample_columns=tuple(samples.columns),
        source_row_count=source_row_count,
        sample_row_count=len(samples),
        excluded_missing_target_count=source_row_count - len(samples),
        observed_ticker_count=len(set(samples.index.get_level_values("ticker"))),
        signal_date_start=_minimum_date(signal_dates),
        signal_date_end=_maximum_date(signal_dates),
        target_end_date_start=_minimum_date(target_end_dates),
        target_end_date_end=_maximum_date(target_end_dates),
        feature_missing_counts=tuple(
            (column, int(features[column].isna().sum())) for column in features.columns
        ),
        selected_target_summary=selected_summary,
        eligible_ticker_count=sum(state.eligible for state in eligibility.values()),
        eligibility_failure_counts=tuple(sorted(failure_counts.items())),
    )


def _validate_bundle_frames(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    samples: pd.DataFrame,
    manifest: TemporalDatasetManifest,
) -> None:
    for frame in (features, targets, samples):
        validate_market_price_index(frame)
    if not features.index.equals(targets.index) or not features.index.equals(samples.index):
        raise ValueError("Bundle frames must use identical sample indexes.")
    if manifest.feature_columns != manifest.spec.feature_set.feature_columns:
        raise ValueError("Manifest feature columns do not match the dataset specification.")
    if manifest.target_columns != manifest.spec.target_set.target_columns:
        raise ValueError("Manifest target columns do not match the dataset specification.")
    if manifest.sample_columns != SAMPLE_METADATA_COLUMNS:
        raise ValueError("Manifest sample columns do not match the canonical metadata schema.")
    if tuple(features.columns) != manifest.feature_columns:
        raise ValueError("Feature columns do not match the bundle manifest.")
    if tuple(targets.columns) != manifest.target_columns:
        raise ValueError("Target columns do not match the bundle manifest.")
    if tuple(samples.columns) != manifest.sample_columns:
        raise ValueError("Sample columns do not match the bundle manifest.")
    if len(features) != manifest.sample_row_count:
        raise ValueError("Bundle row count does not match the manifest.")
    if manifest.source_row_count != (
        manifest.sample_row_count + manifest.excluded_missing_target_count
    ):
        raise ValueError("Manifest source and excluded row counts are inconsistent.")
    if targets[manifest.spec.task.target_column].isna().any():
        raise ValueError("The selected supervised target must be complete.")
    _validate_target_dates(samples, data_cutoff=manifest.spec.data_cutoff)


def _selected_target_summary(
    target: pd.Series,
    *,
    task_type: str,
) -> tuple[tuple[str, int | float], ...]:
    if task_type == "classification":
        counts = Counter(_manifest_scalar(value) for value in target)
        return tuple((f"class:{name}", count) for name, count in sorted(counts.items()))

    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Regression targets must contain only numeric values.")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ValueError("Regression targets must contain only finite values.")
    return (
        ("count", int(numeric.count())),
        ("mean", float(numeric.mean())),
        ("standard_deviation", float(numeric.std(ddof=0))),
        ("minimum", float(numeric.min())),
        ("maximum", float(numeric.max())),
    )


def _manifest_scalar(value: object) -> str:
    text = str(value)
    if text in {"True", "False"}:
        return text.lower()
    return text


def _date_range(start: date | None, end: date | None) -> dict[str, str | None]:
    return {
        "start": start.isoformat() if start is not None else None,
        "end": end.isoformat() if end is not None else None,
    }


def _minimum_date(values: pd.DatetimeIndex) -> date | None:
    return None if values.empty else values.min().date()


def _maximum_date(values: pd.DatetimeIndex) -> date | None:
    return None if values.empty else values.max().date()
