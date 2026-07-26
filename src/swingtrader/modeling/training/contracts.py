"""Runtime contracts shared by baseline training and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

TARGET_COLUMN = "target"
SCORE_COLUMN = "score"
PREDICTED_CLASS_COLUMN = "predicted_class"
SPLIT_COLUMN = "split"
RANKING_RETURN_COLUMN = "ranking_return"
PREDICTION_COLUMNS = (
    SPLIT_COLUMN,
    TARGET_COLUMN,
    SCORE_COLUMN,
    PREDICTED_CLASS_COLUMN,
    RANKING_RETURN_COLUMN,
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Configure deterministic classification, calibration, and ranking evaluation.

    Attributes:
        classification_threshold: Probability cutoff used to derive predicted classes.
        calibration_bins: Number of fixed-width probability buckets.
        score_quantiles: Number of within-date score groups used for ranking summaries.
        top_k: Maximum candidates selected per trading date.
        random_seed: Seed for the date-matched random comparator and deterministic
            score-tie resolution.
    """

    classification_threshold: float = 0.5
    calibration_bins: int = 10
    score_quantiles: int = 10
    top_k: int = 5
    random_seed: int = 0

    def __post_init__(self) -> None:
        threshold = _finite_number(
            self.classification_threshold,
            field_name="Classification threshold",
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Classification threshold must be between zero and one.")
        object.__setattr__(self, "classification_threshold", threshold)
        for field_name, value in (
            ("Calibration bins", self.calibration_bins),
            ("Score quantiles", self.score_quantiles),
            ("Top-k size", self.top_k),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer.")
            if value < 1:
                raise ValueError(f"{field_name} must be positive.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("Evaluation random seed must be an integer.")
        if self.random_seed < 0:
            raise ValueError("Evaluation random seed must not be negative.")

    def to_manifest(self) -> dict[str, int | float]:
        """Return the evaluation choices needed to reproduce a report."""
        return {
            "classification_threshold": self.classification_threshold,
            "calibration_bins": self.calibration_bins,
            "score_quantiles": self.score_quantiles,
            "top_k": self.top_k,
            "random_seed": self.random_seed,
        }


def build_prediction_frame(
    *,
    target: pd.Series,
    score: pd.Series | np.ndarray,
    split: str,
    classification_threshold: float = 0.5,
    ranking_return: pd.Series | None = None,
) -> pd.DataFrame:
    """Build the canonical row-aligned prediction frame for one temporal split."""
    if not isinstance(target, pd.Series):
        raise TypeError("Prediction targets must be a pandas Series.")
    if not isinstance(split, str) or not split.strip():
        raise ValueError("Prediction split must be a non-empty string.")
    threshold = _finite_number(classification_threshold, field_name="Classification threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Classification threshold must be between zero and one.")

    numeric_target = pd.to_numeric(target, errors="coerce").astype("float64")
    if numeric_target.isna().any() or not set(numeric_target.unique()).issubset({0.0, 1.0}):
        raise ValueError("Prediction targets must be complete binary values.")

    if isinstance(score, pd.Series):
        if not score.index.equals(target.index):
            raise ValueError("Prediction scores must share the target index.")
        numeric_score = pd.to_numeric(score, errors="coerce").astype("float64")
    elif isinstance(score, np.ndarray):
        if score.ndim != 1 or len(score) != len(target):
            raise ValueError("Prediction scores must be one-dimensional and match targets.")
        numeric_score = pd.Series(score, index=target.index, dtype="float64")
    else:
        raise TypeError("Prediction scores must be a pandas Series or NumPy array.")
    if not np.isfinite(numeric_score.to_numpy()).all():
        raise ValueError("Prediction scores must be finite.")
    if numeric_score.lt(0.0).any() or numeric_score.gt(1.0).any():
        raise ValueError("Prediction scores must be between zero and one.")

    if ranking_return is None:
        numeric_return = pd.Series(np.nan, index=target.index, dtype="float64")
    else:
        if not isinstance(ranking_return, pd.Series):
            raise TypeError("Ranking returns must be a pandas Series when provided.")
        if not ranking_return.index.equals(target.index):
            raise ValueError("Ranking returns must share the target index.")
        numeric_return = pd.to_numeric(ranking_return, errors="coerce").astype("float64")
        invalid = ranking_return.notna() & numeric_return.isna()
        if invalid.any():
            raise ValueError("Ranking returns must be numeric or missing.")
        finite = np.isfinite(numeric_return.to_numpy(dtype="float64", na_value=np.nan))
        if (~finite & numeric_return.notna().to_numpy()).any():
            raise ValueError("Ranking returns must be finite when present.")

    frame = pd.DataFrame(
        {
            SPLIT_COLUMN: split,
            TARGET_COLUMN: numeric_target.astype("int8"),
            SCORE_COLUMN: numeric_score,
            PREDICTED_CLASS_COLUMN: numeric_score.ge(threshold).astype("int8"),
            RANKING_RETURN_COLUMN: numeric_return,
        },
        index=target.index.copy(),
    )
    validate_prediction_frame(frame)
    return frame


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Validate the canonical prediction schema without mutating the input."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Predictions must be a pandas DataFrame.")
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            "Prediction columns must be ordered as: " + ", ".join(PREDICTION_COLUMNS) + "."
        )
    if not frame.index.is_unique:
        raise ValueError("Prediction index must be unique.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Prediction index must be sorted.")
    if frame.empty:
        raise ValueError("Prediction frame must contain at least one row.")
    empty_split = frame[SPLIT_COLUMN].astype(str).str.len().eq(0)
    if frame[SPLIT_COLUMN].isna().any() or empty_split.any():
        raise ValueError("Prediction split values must be non-empty.")
    for column in (TARGET_COLUMN, PREDICTED_CLASS_COLUMN):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not set(numeric.unique()).issubset({0, 1}):
            raise ValueError(f"Prediction column {column!r} must contain binary values.")
    scores = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Prediction scores must be finite values between zero and one.")
    ranking_return = pd.to_numeric(frame[RANKING_RETURN_COLUMN], errors="coerce")
    invalid_return = frame[RANKING_RETURN_COLUMN].notna() & ranking_return.isna()
    finite_return = np.isfinite(ranking_return.to_numpy(dtype="float64", na_value=np.nan))
    if invalid_return.any() or (~finite_return & ranking_return.notna().to_numpy()).any():
        raise ValueError("Prediction ranking returns must be finite or missing.")


def _finite_number(value: Real, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite.")
    return numeric
