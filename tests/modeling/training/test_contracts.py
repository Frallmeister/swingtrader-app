from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.modeling.training import (
    PREDICTION_COLUMNS,
    EvaluationConfig,
    build_prediction_frame,
    validate_prediction_frame,
)


def test_prediction_frame_has_a_stable_ordered_schema() -> None:
    index = pd.MultiIndex.from_tuples(
        [("yfinance", "AAA.ST", pd.Timestamp("2026-01-01"))],
        names=("provider", "ticker", "trading_date"),
    )
    frame = build_prediction_frame(
        target=pd.Series([1], index=index),
        score=pd.Series([0.7], index=index),
        split="validation",
    )
    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert frame.iloc[0]["predicted_class"] == 1
    validate_prediction_frame(frame)


def test_prediction_frame_rejects_unsorted_or_out_of_range_scores() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", "AAA.ST", pd.Timestamp("2026-01-02")),
            ("yfinance", "AAA.ST", pd.Timestamp("2026-01-01")),
        ],
        names=("provider", "ticker", "trading_date"),
    )
    with pytest.raises(ValueError, match="between zero and one"):
        build_prediction_frame(
            target=pd.Series([0, 1], index=index.sort_values()),
            score=pd.Series([0.1, 1.1], index=index.sort_values()),
            split="validation",
        )


def test_prediction_frame_requires_row_aligned_scores() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", "AAA.ST", pd.Timestamp("2026-01-01")),
            ("yfinance", "BBB.ST", pd.Timestamp("2026-01-01")),
        ],
        names=("provider", "ticker", "trading_date"),
    )
    reversed_index = index[::-1]
    target = pd.Series([0, 1], index=index)

    with pytest.raises(ValueError, match="share the target index"):
        build_prediction_frame(
            target=target,
            score=pd.Series([0.8, 0.2], index=reversed_index),
            split="validation",
        )
    with pytest.raises(ValueError, match="one-dimensional and match targets"):
        build_prediction_frame(
            target=target,
            score=np.array([[0.2, 0.8]]),
            split="validation",
        )


def test_evaluation_config_rejects_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        EvaluationConfig(classification_threshold=1.1)
    with pytest.raises(ValueError, match="Top-k"):
        EvaluationConfig(top_k=0)


def test_evaluation_config_manifest_retains_all_reproducibility_choices() -> None:
    config = EvaluationConfig(
        classification_threshold=0.4,
        calibration_bins=8,
        score_quantiles=5,
        top_k=3,
        random_seed=17,
    )
    assert config.to_manifest() == {
        "classification_threshold": 0.4,
        "calibration_bins": 8,
        "score_quantiles": 5,
        "top_k": 3,
        "random_seed": 17,
    }
