"""Classification, calibration, and daily cross-sectional ranking evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from swingtrader.modeling.training.baselines import deterministic_random_scores
from swingtrader.modeling.training.contracts import (
    PREDICTED_CLASS_COLUMN,
    RANKING_RETURN_COLUMN,
    SCORE_COLUMN,
    SPLIT_COLUMN,
    TARGET_COLUMN,
    EvaluationConfig,
    validate_prediction_frame,
)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Own one split's metrics, provenance, diagnostics, and source predictions."""

    split: str
    config: EvaluationConfig
    ranking_return_column: str | None
    aggregate_metrics: Mapping[str, float]
    dataset_context: Mapping[str, object]
    predictions: pd.DataFrame
    per_date_metrics: pd.DataFrame
    calibration: pd.DataFrame
    score_quantiles: pd.DataFrame
    score_quantiles_by_date: pd.DataFrame
    top_k_by_date: pd.DataFrame
    random_top_k_by_date: pd.DataFrame
    feature_missingness: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError("Evaluation split must be a non-empty string.")
        if not isinstance(self.config, EvaluationConfig):
            raise TypeError("Evaluation report config must be an EvaluationConfig.")
        if self.ranking_return_column is not None and (
            not isinstance(self.ranking_return_column, str)
            or not self.ranking_return_column.strip()
        ):
            raise ValueError("Ranking-return column must be a non-empty string when provided.")
        validate_prediction_frame(self.predictions)
        if set(self.predictions[SPLIT_COLUMN].astype(str).unique()) != {self.split}:
            raise ValueError("Evaluation predictions must belong to the declared split.")
        object.__setattr__(
            self,
            "aggregate_metrics",
            MappingProxyType(dict(self.aggregate_metrics)),
        )
        object.__setattr__(
            self,
            "dataset_context",
            MappingProxyType(dict(self.dataset_context)),
        )
        for field_name in (
            "predictions",
            "per_date_metrics",
            "calibration",
            "score_quantiles",
            "score_quantiles_by_date",
            "top_k_by_date",
            "random_top_k_by_date",
            "feature_missingness",
        ):
            object.__setattr__(self, field_name, getattr(self, field_name).copy(deep=True))

    def to_manifest(self) -> dict[str, object]:
        """Return JSON-compatible pooled results, provenance, and dataset context."""
        return {
            "split": self.split,
            "ranking_return_column": self.ranking_return_column,
            "evaluation_config": self.config.to_manifest(),
            "aggregate_metrics": {
                name: (value if math.isfinite(value) else None)
                for name, value in self.aggregate_metrics.items()
            },
            "dataset_context": dict(self.dataset_context),
            "table_rows": {
                "predictions": len(self.predictions),
                "per_date_metrics": len(self.per_date_metrics),
                "calibration": len(self.calibration),
                "score_quantiles": len(self.score_quantiles),
                "score_quantiles_by_date": len(self.score_quantiles_by_date),
                "top_k_by_date": len(self.top_k_by_date),
                "random_top_k_by_date": len(self.random_top_k_by_date),
                "feature_missingness": len(self.feature_missingness),
            },
        }


def evaluate_predictions(
    predictions: pd.DataFrame,
    *,
    features: pd.DataFrame,
    config: EvaluationConfig | None = None,
    ranking_return_column: str | None = None,
) -> EvaluationReport:
    """Evaluate one prediction frame using pooled and per-date diagnostics.

    ``ranking_return_column`` records the source name for the already-aligned
    ``ranking_return`` values; it does not select or transform outcome data.
    """
    validate_prediction_frame(predictions)
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Evaluation features must be a pandas DataFrame.")
    if not features.index.equals(predictions.index):
        raise ValueError("Evaluation features and predictions must share an index.")
    resolved_config = config or EvaluationConfig()
    if not isinstance(resolved_config, EvaluationConfig):
        raise TypeError("Evaluation requires an EvaluationConfig.")
    expected_class = (
        predictions[SCORE_COLUMN]
        .ge(resolved_config.classification_threshold)
        .astype("int8")
    )
    actual_class = predictions[PREDICTED_CLASS_COLUMN].astype("int8")
    if not expected_class.equals(actual_class):
        raise ValueError(
            "Prediction classes do not match EvaluationConfig.classification_threshold."
        )
    if ranking_return_column is not None and (
        not isinstance(ranking_return_column, str) or not ranking_return_column.strip()
    ):
        raise ValueError("ranking_return_column must be a non-empty string when provided.")
    split_values = predictions[SPLIT_COLUMN].astype(str).unique()
    if len(split_values) != 1:
        raise ValueError("Evaluation predictions must contain exactly one split.")
    split = str(split_values[0])

    aggregate = _classification_metrics(predictions)
    calibration = _calibration_table(predictions, bins=resolved_config.calibration_bins)
    quantiles_by_date, score_quantiles = _score_quantile_tables(
        predictions,
        quantiles=resolved_config.score_quantiles,
        seed=resolved_config.random_seed,
    )
    top_k = _top_k_table(
        predictions,
        top_k=resolved_config.top_k,
        seed=resolved_config.random_seed,
    )
    random_top_k = _random_top_k_table(
        predictions,
        top_k=resolved_config.top_k,
        seed=resolved_config.random_seed,
    )
    per_date = _per_date_metrics(predictions, top_k=top_k, random_top_k=random_top_k)
    aggregate.update(
        _ranking_metrics(
            predictions,
            score_quantiles=score_quantiles,
            top_k=top_k,
            random_top_k=random_top_k,
            quantiles=resolved_config.score_quantiles,
        )
    )
    missingness = _feature_missingness(features)
    context = _dataset_context(predictions, features=features, missingness=missingness)
    return EvaluationReport(
        split=split,
        config=resolved_config,
        ranking_return_column=ranking_return_column,
        aggregate_metrics=aggregate,
        dataset_context=context,
        predictions=predictions,
        per_date_metrics=per_date,
        calibration=calibration,
        score_quantiles=score_quantiles,
        score_quantiles_by_date=quantiles_by_date,
        top_k_by_date=top_k,
        random_top_k_by_date=random_top_k,
        feature_missingness=missingness,
    )


def _classification_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame[TARGET_COLUMN].to_numpy(dtype="int8")
    score = frame[SCORE_COLUMN].to_numpy(dtype="float64")
    predicted = frame[PREDICTED_CLASS_COLUMN].to_numpy(dtype="int8")
    positive_count = int(target.sum())
    negative_count = len(target) - positive_count
    pr_auc = math.nan
    average_precision = math.nan
    if positive_count:
        precision, recall, _ = precision_recall_curve(target, score)
        pr_auc = float(auc(recall, precision))
        average_precision = float(average_precision_score(target, score))
    return {
        "pr_auc": pr_auc,
        "average_precision": average_precision,
        "roc_auc": (
            float(roc_auc_score(target, score)) if positive_count and negative_count else math.nan
        ),
        "log_loss": float(log_loss(target, score, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, score)),
        "precision": float(precision_score(target, predicted, zero_division=0.0)),
        "recall": (
            float(recall_score(target, predicted, zero_division=0.0))
            if positive_count
            else math.nan
        ),
        "prevalence": float(target.mean()),
        "row_count": float(len(frame)),
    }


def _calibration_table(frame: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    scores = frame[SCORE_COLUMN].to_numpy(dtype="float64")
    bucket = np.minimum((scores * bins).astype("int64"), bins - 1) + 1
    working = frame.assign(calibration_bin=bucket)
    grouped = working.groupby("calibration_bin", sort=True, observed=True)
    rows = []
    for bin_number in range(1, bins + 1):
        group = grouped.get_group(bin_number) if bin_number in grouped.groups else working.iloc[0:0]
        rows.append(
            {
                "calibration_bin": bin_number,
                "lower_bound": (bin_number - 1) / bins,
                "upper_bound": bin_number / bins,
                "sample_count": len(group),
                "mean_score": (float(group[SCORE_COLUMN].mean()) if len(group) else math.nan),
                "observed_rate": (float(group[TARGET_COLUMN].mean()) if len(group) else math.nan),
            }
        )
    return pd.DataFrame(rows)


def _score_quantile_tables(
    frame: pd.DataFrame,
    *,
    quantiles: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = frame.copy()
    working["__trading_date"] = _trading_dates(working)
    working["__tie_breaker"] = deterministic_random_scores(working.index, seed=seed)
    quantile_assignments = pd.Series(0, index=working.index, dtype="int64")
    for _, group in working.groupby("__trading_date", sort=True):
        quantile_assignments.loc[group.index] = _quantile_numbers(
            group[SCORE_COLUMN],
            tie_breaker=group["__tie_breaker"],
            quantiles=quantiles,
        )
    working["score_quantile"] = quantile_assignments

    date_rows = []
    grouped_by_date = working.groupby(
        ["__trading_date", "score_quantile"],
        sort=True,
        observed=True,
    )
    for (trading_date, quantile), group in grouped_by_date:
        valid_return = group[RANKING_RETURN_COLUMN].dropna()
        date_rows.append(
            {
                "trading_date": trading_date,
                "score_quantile": int(quantile),
                "sample_count": len(group),
                "mean_score": float(group[SCORE_COLUMN].mean()),
                "positive_rate": float(group[TARGET_COLUMN].mean()),
                "mean_ranking_return": (
                    float(valid_return.mean()) if not valid_return.empty else math.nan
                ),
            }
        )
    by_date = pd.DataFrame(date_rows)

    aggregate_rows = []
    grouped_by_quantile = by_date.groupby("score_quantile", sort=True, observed=True)
    for quantile in range(1, quantiles + 1):
        group = (
            grouped_by_quantile.get_group(quantile)
            if quantile in grouped_by_quantile.groups
            else by_date.iloc[0:0]
        )
        aggregate_rows.append(
            {
                "score_quantile": quantile,
                "sample_count": int(group["sample_count"].sum()) if len(group) else 0,
                "date_count": int(group["trading_date"].nunique()) if len(group) else 0,
                "mean_score": _mean_or_nan(group["mean_score"]),
                "positive_rate": _mean_or_nan(group["positive_rate"]),
                "mean_ranking_return": _mean_or_nan(group["mean_ranking_return"]),
            }
        )
    return by_date, pd.DataFrame(aggregate_rows)


def _top_k_table(frame: pd.DataFrame, *, top_k: int, seed: int) -> pd.DataFrame:
    working = frame.copy()
    working["__trading_date"] = _trading_dates(working)
    working["__tie_breaker"] = deterministic_random_scores(working.index, seed=seed)
    rows = []
    for trading_date, group in working.groupby("__trading_date", sort=True):
        selected = group.sort_values(
            [SCORE_COLUMN, "__tie_breaker"],
            ascending=[False, False],
            kind="mergesort",
        ).head(top_k)
        valid_return = selected[RANKING_RETURN_COLUMN].dropna()
        rows.append(
            {
                "trading_date": trading_date,
                "selected_count": len(selected),
                "positive_rate": float(selected[TARGET_COLUMN].mean()),
                "mean_score": float(selected[SCORE_COLUMN].mean()),
                "ranking_return_count": len(valid_return),
                "mean_ranking_return": (
                    float(valid_return.mean()) if not valid_return.empty else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _random_top_k_table(frame: pd.DataFrame, *, top_k: int, seed: int) -> pd.DataFrame:
    working = frame.copy()
    working["__trading_date"] = _trading_dates(working)
    working["random_score"] = deterministic_random_scores(working.index, seed=seed)
    rows = []
    for trading_date, group in working.groupby("__trading_date", sort=True):
        selected = group.sort_values(
            "random_score",
            ascending=False,
            kind="mergesort",
        ).head(top_k)
        valid_return = selected[RANKING_RETURN_COLUMN].dropna()
        rows.append(
            {
                "trading_date": trading_date,
                "selected_count": len(selected),
                "positive_rate": float(selected[TARGET_COLUMN].mean()),
                "ranking_return_count": len(valid_return),
                "mean_ranking_return": (
                    float(valid_return.mean()) if not valid_return.empty else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _per_date_metrics(
    frame: pd.DataFrame,
    *,
    top_k: pd.DataFrame,
    random_top_k: pd.DataFrame,
) -> pd.DataFrame:
    working = frame.copy()
    working["__trading_date"] = _trading_dates(working)
    top_by_date = top_k.set_index("trading_date")
    random_by_date = random_top_k.set_index("trading_date")
    rows: list[dict[str, object]] = []
    for trading_date, group in working.groupby("__trading_date", sort=True):
        metrics = _classification_metrics(group)
        rows.append(
            {
                "trading_date": trading_date,
                **metrics,
                "spearman": _spearman(group),
                "top_k_count": int(top_by_date.loc[trading_date, "selected_count"]),
                "top_k_positive_rate": float(top_by_date.loc[trading_date, "positive_rate"]),
                "top_k_return_count": int(top_by_date.loc[trading_date, "ranking_return_count"]),
                "top_k_mean_return": float(top_by_date.loc[trading_date, "mean_ranking_return"]),
                "random_top_k_positive_rate": float(
                    random_by_date.loc[trading_date, "positive_rate"]
                ),
                "random_top_k_return_count": int(
                    random_by_date.loc[trading_date, "ranking_return_count"]
                ),
                "random_top_k_mean_return": float(
                    random_by_date.loc[trading_date, "mean_ranking_return"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _ranking_metrics(
    frame: pd.DataFrame,
    *,
    score_quantiles: pd.DataFrame,
    top_k: pd.DataFrame,
    random_top_k: pd.DataFrame,
    quantiles: int,
) -> dict[str, float]:
    working = frame.copy()
    working["__trading_date"] = _trading_dates(working)
    correlations: list[float] = []
    for _, group in working.groupby("__trading_date", sort=True):
        correlation = _spearman(group)
        if math.isfinite(correlation):
            correlations.append(correlation)
    top_quantile = score_quantiles.loc[score_quantiles["score_quantile"].eq(quantiles)].iloc[0]
    paired_returns = _paired_top_k_returns(top_k, random_top_k)
    model_return = _mean_or_nan(paired_returns["model_return"])
    random_return = _mean_or_nan(paired_returns["random_return"])
    model_positive_rate = _mean_or_nan(top_k["positive_rate"])
    random_positive_rate = _mean_or_nan(random_top_k["positive_rate"])
    return {
        "mean_daily_spearman": float(np.mean(correlations)) if correlations else math.nan,
        "mean_top_k_return": model_return,
        "mean_random_top_k_return": random_return,
        "top_k_return_lift": _difference_or_nan(model_return, random_return),
        "top_k_return_comparison_date_count": float(len(paired_returns)),
        "mean_top_k_positive_rate": model_positive_rate,
        "mean_random_top_k_positive_rate": random_positive_rate,
        "top_k_positive_rate_lift": model_positive_rate - random_positive_rate,
        "top_quantile_return": float(top_quantile["mean_ranking_return"]),
        "top_quantile_positive_rate": float(top_quantile["positive_rate"]),
    }


def _paired_top_k_returns(top_k: pd.DataFrame, random_top_k: pd.DataFrame) -> pd.DataFrame:
    paired = top_k[["trading_date", "mean_ranking_return"]].merge(
        random_top_k[["trading_date", "mean_ranking_return"]],
        on="trading_date",
        how="inner",
        validate="one_to_one",
        suffixes=("_model", "_random"),
    )
    paired = paired.dropna(subset=["mean_ranking_return_model", "mean_ranking_return_random"])
    return paired.rename(
        columns={
            "mean_ranking_return_model": "model_return",
            "mean_ranking_return_random": "random_return",
        }
    ).reset_index(drop=True)


def _dataset_context(
    frame: pd.DataFrame,
    *,
    features: pd.DataFrame,
    missingness: pd.DataFrame,
) -> dict[str, object]:
    dates = _trading_dates(frame)
    missing_cells = int(missingness["missing_count"].sum())
    total_feature_cells = int(features.shape[0] * features.shape[1])
    return {
        "row_count": len(frame),
        "trading_date_count": int(dates.nunique()),
        "trading_date_start": dates.min().date().isoformat(),
        "trading_date_end": dates.max().date().isoformat(),
        "ticker_count": int(_ticker_values(frame).nunique()),
        "feature_count": features.shape[1],
        "features_with_missing_values": int(missingness["missing_count"].gt(0).sum()),
        "feature_cell_missing_fraction": (
            missing_cells / total_feature_cells if total_feature_cells else 0.0
        ),
        "prevalence": float(frame[TARGET_COLUMN].mean()),
        "ranking_return_coverage": float(frame[RANKING_RETURN_COLUMN].notna().mean()),
    }


def _feature_missingness(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in features.columns:
        numeric = pd.to_numeric(features[column], errors="coerce")
        values = numeric.to_numpy(dtype="float64", na_value=np.nan)
        missing = numeric.isna() | ~np.isfinite(values)
        rows.append(
            {
                "feature": str(column),
                "missing_count": int(missing.sum()),
                "missing_fraction": float(missing.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_fraction", "feature"],
        ascending=[False, True],
        ignore_index=True,
    )


def _quantile_numbers(
    scores: pd.Series,
    *,
    tie_breaker: pd.Series,
    quantiles: int,
) -> pd.Series:
    order = np.lexsort(
        (
            tie_breaker.to_numpy(dtype="float64"),
            scores.to_numpy(dtype="float64"),
        )
    )
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1, dtype="float64")
    if len(scores) == 1:
        values = np.array([quantiles], dtype="int64")
    else:
        values = 1 + np.floor((ranks - 1) * (quantiles - 1) / (len(scores) - 1))
        values = values.astype("int64")
    return pd.Series(values, index=scores.index)


def _spearman(frame: pd.DataFrame) -> float:
    valid = frame[[SCORE_COLUMN, RANKING_RETURN_COLUMN]].dropna()
    if (
        len(valid) < 2
        or valid[SCORE_COLUMN].nunique() < 2
        or valid[RANKING_RETURN_COLUMN].nunique() < 2
    ):
        return math.nan
    return float(spearmanr(valid[SCORE_COLUMN], valid[RANKING_RETURN_COLUMN]).statistic)


def _mean_or_nan(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else math.nan


def _difference_or_nan(first: float, second: float) -> float:
    return first - second if math.isfinite(first) and math.isfinite(second) else math.nan


def _trading_dates(frame: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(frame.index, pd.MultiIndex) and "trading_date" in frame.index.names:
        return pd.DatetimeIndex(frame.index.get_level_values("trading_date"))
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    raise ValueError("Prediction index must expose a trading_date level or be a DatetimeIndex.")


def _ticker_values(frame: pd.DataFrame) -> pd.Index:
    if isinstance(frame.index, pd.MultiIndex) and "ticker" in frame.index.names:
        return frame.index.get_level_values("ticker")
    return pd.Index(["__single_series"] * len(frame))
