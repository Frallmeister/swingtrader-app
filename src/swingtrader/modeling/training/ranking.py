"""Small helpers for cross-sectional ranking studies in notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

_CANONICAL_INDEX_NAMES = ["provider", "ticker", "trading_date"]
_QUERY_LEVELS = ["provider", "trading_date"]


def prepare_xgboost_ranking_data(
    features: pd.DataFrame,
    relevance: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Sort one split into contiguous date groups and return XGBoost query IDs."""
    _validate_aligned_index(features, relevance)
    numeric_relevance = pd.to_numeric(relevance, errors="coerce")
    values = numeric_relevance.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("Ranking relevance labels must be complete and finite.")
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Ranking relevance labels must be non-negative integers.")

    order = features.index.to_frame(index=False).sort_values(
        ["provider", "trading_date", "ticker"],
        kind="stable",
    ).index.to_numpy()
    sorted_features = features.iloc[order].copy()
    sorted_relevance = numeric_relevance.iloc[order].astype("int64").copy()
    query_index = pd.MultiIndex.from_frame(
        sorted_features.index.to_frame(index=False).loc[:, _QUERY_LEVELS]
    )
    query_ids = pd.factorize(query_index, sort=False)[0].astype("int32")
    return sorted_features, sorted_relevance, query_ids


def evaluate_cross_sectional_scores(
    scores: pd.Series,
    relevance: pd.Series,
    ranking_return: pd.Series,
    *,
    top_k: int = 10,
) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate scores independently within each provider and trading date."""
    _validate_aligned_series(scores, relevance, ranking_return)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    frame = pd.DataFrame(
        {
            "score": pd.to_numeric(scores, errors="coerce"),
            "relevance": pd.to_numeric(relevance, errors="coerce"),
            "ranking_return": pd.to_numeric(ranking_return, errors="coerce"),
        }
    )
    if not np.isfinite(frame.to_numpy(dtype="float64")).all():
        raise ValueError("Ranking evaluation inputs must be complete and finite.")

    rows: list[dict[str, object]] = []
    for (provider, trading_date), group in frame.groupby(
        level=_QUERY_LEVELS,
        sort=False,
    ):
        selected_count = min(top_k, len(group))
        ordered = group.sort_values("score", ascending=False, kind="stable")
        selected = ordered.head(selected_count)
        rows.append(
            {
                "provider": provider,
                "trading_date": trading_date,
                "stock_count": len(group),
                "selected_count": selected_count,
                "ndcg_at_k": float(
                    ndcg_score(
                        group[["relevance"]].to_numpy(dtype="float64").T,
                        group[["score"]].to_numpy(dtype="float64").T,
                        k=selected_count,
                    )
                ),
                "rank_ic": group["score"].corr(
                    group["ranking_return"],
                    method="spearman",
                ),
                "top_k_mean_return": float(selected["ranking_return"].mean()),
                "universe_mean_return": float(group["ranking_return"].mean()),
            }
        )

    daily = pd.DataFrame(rows).set_index(_QUERY_LEVELS).sort_index()
    daily["top_k_excess_return"] = (
        daily["top_k_mean_return"] - daily["universe_mean_return"]
    )
    valid_rank_ic = daily["rank_ic"].dropna()
    summary = pd.Series(
        {
            "date_count": float(len(daily)),
            "mean_ndcg_at_k": float(daily["ndcg_at_k"].mean()),
            "mean_rank_ic": float(valid_rank_ic.mean()),
            "positive_rank_ic_fraction": float(valid_rank_ic.gt(0).mean()),
            "mean_top_k_return": float(daily["top_k_mean_return"].mean()),
            "mean_top_k_excess_return": float(daily["top_k_excess_return"].mean()),
        },
        dtype="float64",
    )
    return summary, daily


def _validate_aligned_index(features: pd.DataFrame, relevance: pd.Series) -> None:
    if list(features.index.names) != _CANONICAL_INDEX_NAMES:
        raise ValueError(
            "Ranking data must use index levels provider, ticker, and trading_date."
        )
    if not features.index.is_unique:
        raise ValueError("Ranking data index must be unique.")
    if not features.index.equals(relevance.index):
        raise ValueError("Ranking features and relevance labels must use identical indexes.")


def _validate_aligned_series(*series: pd.Series) -> None:
    first = series[0]
    if list(first.index.names) != _CANONICAL_INDEX_NAMES:
        raise ValueError(
            "Ranking data must use index levels provider, ticker, and trading_date."
        )
    if not first.index.is_unique:
        raise ValueError("Ranking data index must be unique.")
    if any(not first.index.equals(item.index) for item in series[1:]):
        raise ValueError("Ranking evaluation inputs must use identical indexes.")
