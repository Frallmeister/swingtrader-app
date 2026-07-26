"""Deterministic baseline models for binary temporal ranking experiments."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
    """Retain train-fitted scikit-learn imputation and scaling state.

    Missing and infinite values are treated as missing, imputed with training
    medians, then mean-centered and scaled to unit population variance. Columns
    that are entirely missing during fitting are retained and imputed with zero.
    """

    columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler

    def __post_init__(self) -> None:
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Preprocessor columns must be non-empty and unique.")
        if any(not isinstance(column, str) or not column for column in self.columns):
            raise ValueError("Preprocessor columns must be non-empty strings.")
        if not isinstance(self.imputer, SimpleImputer):
            raise TypeError("Preprocessor imputer must be a fitted SimpleImputer.")
        if not isinstance(self.scaler, StandardScaler):
            raise TypeError("Preprocessor scaler must be a fitted StandardScaler.")
        if not hasattr(self.imputer, "statistics_") or not hasattr(self.scaler, "scale_"):
            raise ValueError("Preprocessor estimators must be fitted.")
        medians = np.asarray(self.imputer.statistics_, dtype="float64")
        means = np.asarray(self.scaler.mean_, dtype="float64")
        scales = np.asarray(self.scaler.scale_, dtype="float64")
        if medians.shape != (len(self.columns),):
            raise ValueError("Imputer state must align with preprocessor columns.")
        if means.shape != (len(self.columns),) or scales.shape != (len(self.columns),):
            raise ValueError("Scaler state must align with preprocessor columns.")
        if not np.isfinite(medians).all() or not np.isfinite(means).all():
            raise ValueError("Preprocessor location statistics must be finite.")
        if not np.isfinite(scales).all() or (scales <= 0.0).any():
            raise ValueError("Preprocessor scales must be positive and finite.")

    @property
    def medians(self) -> tuple[float, ...]:
        """Return fitted median-imputation values in feature-column order."""
        return tuple(float(value) for value in self.imputer.statistics_)

    @property
    def means(self) -> tuple[float, ...]:
        """Return fitted post-imputation means in feature-column order."""
        return tuple(float(value) for value in self.scaler.mean_)

    @property
    def scales(self) -> tuple[float, ...]:
        """Return fitted population standard deviations in feature-column order."""
        return tuple(float(value) for value in self.scaler.scale_)

    @classmethod
    def fit(cls, features: pd.DataFrame) -> MedianStandardizer:
        """Fit median imputation and standardization on training rows only."""
        numeric = _numeric_features(features).replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = imputer.fit_transform(numeric)
        scaler = StandardScaler()
        scaler.fit(imputed)
        return cls(
            columns=tuple(str(column) for column in numeric.columns),
            imputer=imputer,
            scaler=scaler,
        )

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the frozen median imputation and scaling to another frame."""
        numeric = _numeric_features(features)
        observed = tuple(str(column) for column in numeric.columns)
        if observed != self.columns:
            raise ValueError("Feature columns or order do not match the fitted preprocessor.")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        imputed = self.imputer.transform(numeric)
        return np.asarray(self.scaler.transform(imputed), dtype="float64")

    def to_manifest(self) -> dict[str, object]:
        """Return the retained preprocessing state in deterministic column order."""
        return {
            "type": "median_imputer_standard_scaler",
            "implementation": "scikit-learn",
            "columns": list(self.columns),
            "medians": dict(zip(self.columns, self.medians, strict=True)),
            "means": dict(zip(self.columns, self.means, strict=True)),
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
    """L2-regularized scikit-learn logistic model with frozen preprocessing."""

    preprocessor: MedianStandardizer
    estimator: LogisticRegression
    regularization_strength: float
    objective: float
    training_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.preprocessor, MedianStandardizer):
            raise TypeError("Logistic preprocessing must be a MedianStandardizer.")
        if not isinstance(self.estimator, LogisticRegression):
            raise TypeError("Logistic estimator must be a fitted LogisticRegression.")
        if not hasattr(self.estimator, "coef_") or not hasattr(self.estimator, "classes_"):
            raise ValueError("Logistic estimator must be fitted.")
        if not np.array_equal(self.estimator.classes_, np.asarray([0, 1])):
            raise ValueError("Logistic estimator must represent binary classes zero and one.")
        if self.estimator.coef_.shape != (1, len(self.preprocessor.columns)):
            raise ValueError("Logistic coefficients must align with preprocessor columns.")
        fitted_values = np.asarray(
            (*self.estimator.intercept_, *self.estimator.coef_.ravel(), self.objective),
            dtype="float64",
        )
        if not np.isfinite(fitted_values).all():
            raise ValueError("Logistic fitted state must be finite.")
        _positive_float(self.regularization_strength, name="regularization_strength")
        _validate_training_rows(self.training_rows)

    @property
    def intercept(self) -> float:
        """Return the fitted binary-class intercept."""
        return float(self.estimator.intercept_[0])

    @property
    def coefficients(self) -> tuple[float, ...]:
        """Return fitted coefficients in feature-column order."""
        return tuple(float(value) for value in self.estimator.coef_[0])

    @property
    def iterations(self) -> int:
        """Return the number of solver iterations used during fitting."""
        return int(self.estimator.n_iter_[0])

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """Return positive-class probabilities for the supplied rows."""
        transformed = self.preprocessor.transform(features)
        positive_class = int(np.flatnonzero(self.estimator.classes_ == 1)[0])
        scores = self.estimator.predict_proba(transformed)[:, positive_class]
        return pd.Series(scores, index=features.index, dtype="float64", name="score")

    def to_manifest(self) -> dict[str, object]:
        """Return fitted coefficients, preprocessing, and solver diagnostics."""
        return {
            "model_type": LOGISTIC_REGRESSION_MODEL_TYPE,
            "implementation": "sklearn.linear_model.LogisticRegression",
            "library_version": sklearn.__version__,
            "training_rows": self.training_rows,
            "regularization_strength": self.regularization_strength,
            "C": float(self.estimator.C),
            "regularization_objective": (
                "mean_log_loss + 0.5 * regularization_strength * squared_l2_norm"
            ),
            "solver": self.estimator.solver,
            "regularization": "l2",
            "l1_ratio": float(self.estimator.l1_ratio),
            "max_iter": int(self.estimator.max_iter),
            "tolerance": float(self.estimator.tol),
            "random_seed": self.estimator.random_state,
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
        return _fit_logistic(
            features,
            binary_target,
            hyperparameters=hyperparameters,
            seed=seed,
        )

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
    seed: int,
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
    estimator = LogisticRegression(
        C=1.0 / (regularization_strength * len(target)),
        l1_ratio=0.0,
        solver="lbfgs",
        max_iter=max_iter,
        tol=tolerance,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            estimator.fit(transformed, target.to_numpy(dtype="int8"))
        except ConvergenceWarning as error:
            raise RuntimeError(
                "Logistic-regression optimization did not converge within "
                f"{max_iter} iterations."
            ) from error
    logits = estimator.decision_function(transformed)
    target_values = target.to_numpy(dtype="float64")
    coefficients = estimator.coef_[0]
    objective = np.mean(np.logaddexp(0.0, logits) - target_values * logits)
    objective += 0.5 * regularization_strength * np.dot(coefficients, coefficients)
    return RegularizedLogisticRegression(
        preprocessor=preprocessor,
        estimator=estimator,
        regularization_strength=regularization_strength,
        objective=float(objective),
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
