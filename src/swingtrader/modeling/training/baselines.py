"""Deterministic baseline models for binary temporal ranking experiments."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from swingtrader.modeling.experiments.contracts import ModelSpec

CONSTANT_PRIOR_MODEL_TYPE = "swingtrader.modeling.training.baselines.ConstantPriorClassifier"
RANDOM_RANKING_MODEL_TYPE = "swingtrader.modeling.training.baselines.DateMatchedRandomRanker"
LOGISTIC_REGRESSION_MODEL_TYPE = (
    "swingtrader.modeling.training.baselines.RegularizedLogisticRegression"
)


class BaselineModelArtifact(Protocol):
    """Minimal fitted-model interface consumed by the evaluation harness."""

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return positive-class scores aligned to ``features``."""

    def to_manifest(self) -> dict[str, object]:
        """Return JSON-compatible fitted state and diagnostics."""


@dataclass(frozen=True, slots=True)
class MedianStandardizer:
    """Retain train-fitted numeric imputation and standardization state."""

    columns: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Preprocessor columns must be non-empty and unique.")
        if any(not isinstance(column, str) or not column for column in self.columns):
            raise ValueError("Preprocessor columns must be non-empty strings.")
        if len(self.medians) != len(self.columns) or len(self.scales) != len(self.columns):
            raise ValueError("Preprocessor state must align with its columns.")
        if not np.isfinite(np.asarray(self.medians, dtype="float64")).all():
            raise ValueError("Preprocessor medians must be finite.")
        scales = np.asarray(self.scales, dtype="float64")
        if not np.isfinite(scales).all() or (scales <= 0.0).any():
            raise ValueError("Preprocessor scales must be positive and finite.")

    @classmethod
    def fit(cls, features: pd.DataFrame) -> MedianStandardizer:
        """Fit medians and population standard deviations on training rows only."""
        numeric = _numeric_features(features).replace([np.inf, -np.inf], np.nan)
        medians = numeric.median(axis=0, skipna=True).fillna(0.0).to_numpy(dtype="float64")
        values = numeric.to_numpy(dtype="float64", copy=True)
        imputed = np.where(np.isnan(values), medians, values)
        scales = np.std(imputed, axis=0, ddof=0)
        scales = np.where(np.isfinite(scales) & (scales > 0.0), scales, 1.0)
        return cls(
            columns=tuple(str(column) for column in numeric.columns),
            medians=tuple(float(value) for value in medians),
            scales=tuple(float(value) for value in scales),
        )

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the frozen training transformation to another feature frame."""
        numeric = _numeric_features(features)
        observed = tuple(str(column) for column in numeric.columns)
        if observed != self.columns:
            raise ValueError("Feature columns or order do not match the fitted preprocessor.")
        values = numeric.to_numpy(dtype="float64", copy=True)
        values[~np.isfinite(values)] = np.nan
        medians = np.asarray(self.medians, dtype="float64")
        scales = np.asarray(self.scales, dtype="float64")
        values = np.where(np.isnan(values), medians, values)
        return (values - medians) / scales

    def to_manifest(self) -> dict[str, object]:
        """Return the retained preprocessing state in deterministic column order."""
        return {
            "type": "median_standardizer",
            "columns": list(self.columns),
            "medians": dict(zip(self.columns, self.medians, strict=True)),
            "scales": dict(zip(self.columns, self.scales, strict=True)),
        }


@dataclass(frozen=True, slots=True)
class ConstantPriorClassifier:
    """Assign the training positive-class prevalence to every sample."""

    prior: float
    training_rows: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.prior) or not 0.0 <= self.prior <= 1.0:
            raise ValueError("Fitted class prior must be finite and between zero and one.")
        _validate_training_rows(self.training_rows)

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return the fitted prevalence for every supplied feature row."""
        _require_feature_frame(features)
        return pd.Series(self.prior, index=features.index, dtype="float64", name="score")

    def to_manifest(self) -> dict[str, object]:
        """Return fitted prevalence and training-row count."""
        return {
            "model_type": CONSTANT_PRIOR_MODEL_TYPE,
            "training_rows": self.training_rows,
            "prior": self.prior,
        }


@dataclass(frozen=True, slots=True)
class DateMatchedRandomRanker:
    """Emit stable pseudo-random scores that can be ranked within each date."""

    seed: int
    training_rows: int

    def __post_init__(self) -> None:
        _validate_seed(self.seed)
        _validate_training_rows(self.training_rows)

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return deterministic random scores derived from sample identity."""
        _require_feature_frame(features)
        return deterministic_random_scores(features.index, seed=self.seed)

    def to_manifest(self) -> dict[str, object]:
        """Return the random seed and training-row count."""
        return {
            "model_type": RANDOM_RANKING_MODEL_TYPE,
            "training_rows": self.training_rows,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class RegularizedLogisticRegression:
    """L2-regularized logistic model with retained preprocessing state."""

    preprocessor: MedianStandardizer
    intercept: float
    coefficients: tuple[float, ...]
    regularization_strength: float
    iterations: int
    objective: float
    training_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.preprocessor, MedianStandardizer):
            raise TypeError("Logistic preprocessing must be a MedianStandardizer.")
        if len(self.coefficients) != len(self.preprocessor.columns):
            raise ValueError("Logistic coefficients must align with preprocessor columns.")
        fitted_values = np.asarray(
            (self.intercept, *self.coefficients, self.objective),
            dtype="float64",
        )
        if not np.isfinite(fitted_values).all():
            raise ValueError("Logistic fitted state must be finite.")
        _positive_float(self.regularization_strength, name="regularization_strength")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise TypeError("Logistic iteration count must be an integer.")
        if self.iterations < 0:
            raise ValueError("Logistic iteration count must not be negative.")
        _validate_training_rows(self.training_rows)

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return positive-class probabilities for the supplied rows."""
        transformed = self.preprocessor.transform(features)
        coefficients = np.asarray(self.coefficients, dtype="float64")
        scores = expit(self.intercept + transformed @ coefficients)
        return pd.Series(scores, index=features.index, dtype="float64", name="score")

    def to_manifest(self) -> dict[str, object]:
        """Return fitted coefficients, preprocessing, and optimizer diagnostics."""
        return {
            "model_type": LOGISTIC_REGRESSION_MODEL_TYPE,
            "training_rows": self.training_rows,
            "regularization_strength": self.regularization_strength,
            "iterations": self.iterations,
            "objective": self.objective,
            "intercept": self.intercept,
            "coefficients": dict(zip(self.preprocessor.columns, self.coefficients, strict=True)),
            "preprocessing": self.preprocessor.to_manifest(),
        }


def fit_baseline_model(
    spec: ModelSpec,
    *,
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
) -> BaselineModelArtifact:
    """Fit one repository baseline selected by ``ModelSpec.model_type``."""
    if not isinstance(spec, ModelSpec):
        raise TypeError("Baseline fitting requires a ModelSpec.")
    _require_feature_frame(features)
    binary_target = _binary_target(target, expected_index=features.index)
    _validate_seed(seed)
    hyperparameters = dict(spec.hyperparameters)

    if spec.model_type == CONSTANT_PRIOR_MODEL_TYPE:
        _reject_unknown_hyperparameters(hyperparameters, allowed=frozenset())
        return ConstantPriorClassifier(
            prior=float(binary_target.mean()),
            training_rows=len(binary_target),
        )
    if spec.model_type == RANDOM_RANKING_MODEL_TYPE:
        _reject_unknown_hyperparameters(hyperparameters, allowed=frozenset())
        return DateMatchedRandomRanker(seed=seed, training_rows=len(binary_target))
    if spec.model_type == LOGISTIC_REGRESSION_MODEL_TYPE:
        return _fit_logistic(features, binary_target, hyperparameters=hyperparameters)

    supported = ", ".join(
        (
            CONSTANT_PRIOR_MODEL_TYPE,
            RANDOM_RANKING_MODEL_TYPE,
            LOGISTIC_REGRESSION_MODEL_TYPE,
        )
    )
    raise ValueError(
        f"Unsupported baseline model type {spec.model_type!r}; expected one of: {supported}."
    )


def deterministic_random_scores(index: pd.Index, *, seed: int) -> pd.Series:
    """Return stable uniform scores derived from the seed and index identity."""
    _validate_seed(seed)
    scores = np.fromiter(
        (_score_from_identity(identity, seed=seed) for identity in index),
        dtype="float64",
        count=len(index),
    )
    return pd.Series(scores, index=index, dtype="float64", name="score")


def _fit_logistic(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    hyperparameters: Mapping[str, object],
) -> RegularizedLogisticRegression:
    allowed = frozenset({"regularization_strength", "max_iter", "tolerance"})
    _reject_unknown_hyperparameters(hyperparameters, allowed=allowed)
    regularization_strength = _positive_float(
        hyperparameters.get("regularization_strength", 1.0),
        name="regularization_strength",
    )
    max_iter = _positive_int(hyperparameters.get("max_iter", 1_000), name="max_iter")
    tolerance = _positive_float(hyperparameters.get("tolerance", 1e-8), name="tolerance")
    if target.nunique() < 2:
        raise ValueError(
            "Regularized logistic regression requires both target classes in training."
        )

    preprocessor = MedianStandardizer.fit(features)
    transformed = preprocessor.transform(features)
    y = target.to_numpy(dtype="float64")
    initial = np.zeros(transformed.shape[1] + 1, dtype="float64")
    initial[0] = math.log(float(y.mean()) / (1.0 - float(y.mean())))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        logits = intercept + transformed @ coefficients
        probabilities = expit(logits)
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += 0.5 * regularization_strength * np.dot(coefficients, coefficients)
        residual = probabilities - y
        gradient = np.empty_like(parameters)
        gradient[0] = residual.mean()
        gradient[1:] = transformed.T @ residual / len(y) + regularization_strength * coefficients
        return float(loss), gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tolerance, "gtol": tolerance},
    )
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("Logistic-regression optimization produced non-finite fitted state.")
    if not result.success:
        raise RuntimeError(
            "Logistic-regression optimization did not converge: " + str(result.message)
        )
    return RegularizedLogisticRegression(
        preprocessor=preprocessor,
        intercept=float(result.x[0]),
        coefficients=tuple(float(value) for value in result.x[1:]),
        regularization_strength=regularization_strength,
        iterations=int(result.nit),
        objective=float(result.fun),
        training_rows=len(target),
    )


def _score_from_identity(identity: object, *, seed: int) -> float:
    parts = identity if isinstance(identity, tuple) else (identity,)
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (value + 0.5) / 2**64


def _numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    _require_feature_frame(features)
    numeric = features.apply(pd.to_numeric, errors="coerce").astype("float64")
    invalid = features.notna() & numeric.isna()
    if invalid.any().any():
        columns = ", ".join(str(column) for column in invalid.columns[invalid.any()])
        raise ValueError(f"Feature columns contain non-numeric values: {columns}.")
    return numeric


def _require_feature_frame(features: pd.DataFrame) -> None:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Model features must be a pandas DataFrame.")
    if features.empty:
        raise ValueError("Model features must contain at least one row.")
    if features.shape[1] < 1:
        raise ValueError("Model features must contain at least one column.")
    if not features.index.is_unique:
        raise ValueError("Model feature index must be unique.")
    if not features.columns.is_unique:
        raise ValueError("Model feature columns must be unique.")
    if any(not isinstance(column, str) or not column for column in features.columns):
        raise ValueError("Model feature columns must be non-empty strings.")


def _binary_target(target: pd.Series, *, expected_index: pd.Index) -> pd.Series:
    if not isinstance(target, pd.Series):
        raise TypeError("Model target must be a pandas Series.")
    if not target.index.equals(expected_index):
        raise ValueError("Model target must share the feature index.")
    numeric = pd.to_numeric(target, errors="coerce").astype("float64")
    if numeric.isna().any() or not set(numeric.unique()).issubset({0.0, 1.0}):
        raise ValueError("Model target must contain complete binary values.")
    return numeric


def _reject_unknown_hyperparameters(
    hyperparameters: Mapping[str, object], *, allowed: frozenset[str]
) -> None:
    unknown = sorted(set(hyperparameters).difference(allowed))
    if unknown:
        raise ValueError("Unknown baseline hyperparameters: " + ", ".join(unknown) + ".")


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return value


def _validate_training_rows(training_rows: int) -> None:
    if isinstance(training_rows, bool) or not isinstance(training_rows, int):
        raise TypeError("Training row count must be an integer.")
    if training_rows < 1:
        raise ValueError("Training row count must be positive.")


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Random seed must be an integer.")
    if seed < 0:
        raise ValueError("Random seed must not be negative.")
