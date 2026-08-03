import numpy as np
import pandas as pd
import pytest

from swingtrader.modeling.training.ranking import (
    evaluate_cross_sectional_scores,
    prepare_xgboost_ranking_data,
)


def _index(rows: list[tuple[str, str, str]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [(provider, ticker, pd.Timestamp(date)) for provider, ticker, date in rows],
        names=["provider", "ticker", "trading_date"],
    )


def test_prepare_xgboost_ranking_data_makes_date_groups_contiguous() -> None:
    index = _index(
        [
            ("yfinance", "AAA", "2026-01-02"),
            ("yfinance", "AAA", "2026-01-03"),
            ("yfinance", "BBB", "2026-01-02"),
            ("yfinance", "BBB", "2026-01-03"),
        ]
    )
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]}, index=index)
    relevance = pd.Series([0, 2, 1, 3], index=index)

    sorted_features, sorted_relevance, query_ids = prepare_xgboost_ranking_data(
        features,
        relevance,
    )

    assert sorted_features.index.get_level_values("trading_date").tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-03"),
    ]
    assert sorted_relevance.tolist() == [0, 1, 2, 3]
    assert query_ids.tolist() == [0, 0, 1, 1]


def test_prepare_xgboost_ranking_data_rejects_non_integer_relevance() -> None:
    index = _index([("yfinance", "AAA", "2026-01-02")])
    features = pd.DataFrame({"feature": [1.0]}, index=index)
    relevance = pd.Series([0.5], index=index)

    with pytest.raises(ValueError, match="non-negative integers"):
        prepare_xgboost_ranking_data(features, relevance)


def test_evaluate_cross_sectional_scores_reports_perfect_ranking() -> None:
    index = _index(
        [
            ("yfinance", "AAA", "2026-01-02"),
            ("yfinance", "BBB", "2026-01-02"),
            ("yfinance", "CCC", "2026-01-02"),
            ("yfinance", "AAA", "2026-01-03"),
            ("yfinance", "BBB", "2026-01-03"),
            ("yfinance", "CCC", "2026-01-03"),
        ]
    )
    scores = pd.Series([3, 2, 1, 6, 5, 4], index=index, dtype="float64")
    relevance = pd.Series([2, 1, 0, 2, 1, 0], index=index, dtype="int64")
    ranking_return = pd.Series(
        [0.03, 0.02, 0.01, 0.06, 0.05, 0.04],
        index=index,
    )

    summary, daily = evaluate_cross_sectional_scores(
        scores,
        relevance,
        ranking_return,
        top_k=1,
    )

    assert summary["date_count"] == 2
    assert summary["mean_ndcg_at_k"] == pytest.approx(1.0)
    assert summary["mean_rank_ic"] == pytest.approx(1.0)
    assert summary["positive_rank_ic_fraction"] == pytest.approx(1.0)
    assert summary["mean_top_k_return"] == pytest.approx(0.045)
    assert np.allclose(daily["top_k_excess_return"], [0.01, 0.01])
