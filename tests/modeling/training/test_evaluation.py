from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from swingtrader.modeling.training import (
    EvaluationConfig,
    build_prediction_frame,
    deterministic_random_scores,
    evaluate_predictions,
    write_evaluation_artifacts,
)


def _index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            ("yfinance", ticker, trading_date)
            for ticker in ("AAA.ST", "BBB.ST", "CCC.ST", "DDD.ST")
            for trading_date in pd.date_range("2026-01-01", periods=3)
        ],
        names=("provider", "ticker", "trading_date"),
    )


def test_perfect_predictions_produce_expected_classification_and_ranking_metrics() -> None:
    index = _index()
    ticker = index.get_level_values("ticker")
    target = pd.Series(ticker.isin(["CCC.ST", "DDD.ST"]).astype(int), index=index)
    score = pd.Series(
        ticker.map({"AAA.ST": 0.05, "BBB.ST": 0.2, "CCC.ST": 0.8, "DDD.ST": 0.95}),
        index=index,
    )
    ranking_return = pd.Series(
        ticker.map({"AAA.ST": -0.02, "BBB.ST": -0.01, "CCC.ST": 0.01, "DDD.ST": 0.03}),
        index=index,
    )
    predictions = build_prediction_frame(
        target=target,
        score=score,
        split="validation",
        ranking_return=ranking_return,
    )
    features = pd.DataFrame(
        {"feature_a": np.arange(len(index), dtype=float), "feature_b": np.nan},
        index=index,
    )
    config = EvaluationConfig(
        calibration_bins=5,
        score_quantiles=4,
        top_k=2,
        random_seed=7,
    )
    report = evaluate_predictions(predictions, features=features, config=config)

    assert report.aggregate_metrics["roc_auc"] == pytest.approx(1.0)
    assert report.aggregate_metrics["average_precision"] == pytest.approx(1.0)
    assert report.aggregate_metrics["precision"] == pytest.approx(1.0)
    assert report.aggregate_metrics["recall"] == pytest.approx(1.0)
    assert report.aggregate_metrics["mean_daily_spearman"] == pytest.approx(1.0)
    assert report.aggregate_metrics["top_quantile_positive_rate"] == pytest.approx(1.0)
    assert report.top_k_by_date["selected_count"].eq(2).all()
    assert report.random_top_k_by_date["selected_count"].eq(2).all()
    assert len(report.per_date_metrics) == 3
    assert len(report.score_quantiles_by_date) == 12
    assert report.score_quantiles["date_count"].eq(3).all()
    assert report.feature_missingness.iloc[0]["feature"] == "feature_b"
    assert report.feature_missingness.iloc[0]["missing_fraction"] == pytest.approx(1.0)
    assert report.dataset_context["trading_date_start"] == "2026-01-01"
    assert report.dataset_context["trading_date_end"] == "2026-01-03"


def test_random_comparison_is_deterministic_and_date_matched() -> None:
    index = _index()
    predictions = build_prediction_frame(
        target=pd.Series(
            index.get_level_values("ticker").isin(["BBB.ST", "DDD.ST"]).astype(int),
            index=index,
        ),
        score=pd.Series(np.linspace(0.01, 0.99, len(index)), index=index),
        split="validation",
        ranking_return=pd.Series(np.linspace(-0.03, 0.03, len(index)), index=index),
    )
    features = pd.DataFrame({"feature": 1.0}, index=index)
    config = EvaluationConfig(top_k=3, random_seed=17)
    first = evaluate_predictions(predictions, features=features, config=config)
    second = evaluate_predictions(predictions, features=features, config=config)

    pd.testing.assert_frame_equal(first.random_top_k_by_date, second.random_top_k_by_date)
    pd.testing.assert_series_equal(
        first.top_k_by_date["selected_count"],
        first.random_top_k_by_date["selected_count"],
        check_names=False,
    )


def test_equal_scores_use_the_random_seed_instead_of_index_order() -> None:
    index = _index()
    ticker = index.get_level_values("ticker")
    target = pd.Series(ticker.isin(["BBB.ST", "DDD.ST"]).astype(int), index=index)
    ranking_return = pd.Series(
        ticker.map({"AAA.ST": -0.02, "BBB.ST": 0.01, "CCC.ST": -0.01, "DDD.ST": 0.03}),
        index=index,
    )
    config = EvaluationConfig(score_quantiles=4, top_k=2, random_seed=7)
    tied_predictions = build_prediction_frame(
        target=target,
        score=pd.Series(0.5, index=index),
        split="validation",
        ranking_return=ranking_return,
    )
    tied_report = evaluate_predictions(
        tied_predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
        config=config,
    )

    common_top_k_columns = [
        "trading_date",
        "selected_count",
        "positive_rate",
        "ranking_return_count",
        "mean_ranking_return",
    ]
    pd.testing.assert_frame_equal(
        tied_report.top_k_by_date[common_top_k_columns],
        tied_report.random_top_k_by_date[common_top_k_columns],
    )

    random_predictions = build_prediction_frame(
        target=target,
        score=deterministic_random_scores(index, seed=7),
        split="validation",
        ranking_return=ranking_return,
    )
    random_report = evaluate_predictions(
        random_predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
        config=config,
    )
    quantile_outcome_columns = [
        "score_quantile",
        "sample_count",
        "date_count",
        "positive_rate",
        "mean_ranking_return",
    ]
    pd.testing.assert_frame_equal(
        tied_report.score_quantiles[quantile_outcome_columns],
        random_report.score_quantiles[quantile_outcome_columns],
    )


def test_top_k_return_comparison_uses_only_paired_dates() -> None:
    dates = pd.date_range("2026-01-01", periods=3)
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", ticker, trading_date)
            for ticker in ("AAA.ST", "BBB.ST")
            for trading_date in dates
        ],
        names=("provider", "ticker", "trading_date"),
    )
    score = pd.Series(
        index.get_level_values("ticker").map({"AAA.ST": 0.9, "BBB.ST": 0.1}),
        index=index,
    )
    target = pd.Series([1, 0, 1, 0, 1, 0], index=index)
    ranking_return = pd.Series([0.10, np.nan, 0.03, np.nan, -0.10, -0.02], index=index)
    config = EvaluationConfig(top_k=1, random_seed=9)

    unpaired_predictions = build_prediction_frame(
        target=target[index.get_level_values("trading_date") < dates[2]],
        score=score[index.get_level_values("trading_date") < dates[2]],
        split="validation",
        ranking_return=ranking_return[index.get_level_values("trading_date") < dates[2]],
    )
    unpaired_report = evaluate_predictions(
        unpaired_predictions,
        features=pd.DataFrame({"feature": 1.0}, index=unpaired_predictions.index),
        config=config,
    )
    assert unpaired_report.aggregate_metrics["top_k_return_comparison_date_count"] == 0
    assert np.isnan(unpaired_report.aggregate_metrics["mean_top_k_return"])
    assert np.isnan(unpaired_report.aggregate_metrics["mean_random_top_k_return"])
    assert np.isnan(unpaired_report.aggregate_metrics["top_k_return_lift"])

    predictions = build_prediction_frame(
        target=target,
        score=score,
        split="validation",
        ranking_return=ranking_return,
    )
    report = evaluate_predictions(
        predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
        config=config,
    )
    assert report.aggregate_metrics["top_k_return_comparison_date_count"] == 1
    assert report.aggregate_metrics["mean_top_k_return"] == pytest.approx(0.03)
    assert report.aggregate_metrics["mean_random_top_k_return"] == pytest.approx(-0.02)
    assert report.aggregate_metrics["top_k_return_lift"] == pytest.approx(0.05)


def test_small_daily_universe_still_populates_the_top_score_quantile() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", ticker, trading_date)
            for ticker in ("AAA.ST", "BBB.ST")
            for trading_date in pd.date_range("2026-01-01", periods=2)
        ],
        names=("provider", "ticker", "trading_date"),
    )
    predictions = build_prediction_frame(
        target=pd.Series([0, 1, 0, 1], index=index),
        score=pd.Series([0.1, 0.2, 0.8, 0.9], index=index),
        split="validation",
        ranking_return=pd.Series([-0.01, -0.02, 0.01, 0.02], index=index),
    )
    report = evaluate_predictions(
        predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
        config=EvaluationConfig(score_quantiles=10, top_k=1),
    )

    top_quantile = report.score_quantiles.loc[report.score_quantiles["score_quantile"].eq(10)].iloc[
        0
    ]
    assert top_quantile["date_count"] == 2
    assert top_quantile["sample_count"] == 2


def test_one_class_dates_keep_undefined_auc_in_per_date_results() -> None:
    index = _index()
    dates = index.get_level_values("trading_date")
    target_values = []
    for ticker, trading_date in zip(
        index.get_level_values("ticker"),
        dates,
        strict=True,
    ):
        if trading_date == pd.Timestamp("2026-01-01"):
            target_values.append(0)
        elif trading_date == pd.Timestamp("2026-01-02"):
            target_values.append(1)
        else:
            target_values.append(int(ticker in {"BBB.ST", "DDD.ST"}))
    predictions = build_prediction_frame(
        target=pd.Series(target_values, index=index),
        score=pd.Series(np.linspace(0.1, 0.9, len(index)), index=index),
        split="validation",
    )
    report = evaluate_predictions(
        predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
    )

    assert np.isnan(report.per_date_metrics.loc[0, "roc_auc"])
    assert np.isnan(report.per_date_metrics.loc[1, "roc_auc"])
    assert report.dataset_context["ranking_return_coverage"] == 0.0


def test_average_precision_groups_equal_scores_as_one_threshold() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("yfinance", "AAA.ST", pd.Timestamp("2026-01-02")),
            ("yfinance", "BBB.ST", pd.Timestamp("2026-01-02")),
        ],
        names=("provider", "ticker", "trading_date"),
    )
    predictions = build_prediction_frame(
        target=pd.Series([1, 0], index=index),
        score=pd.Series([0.5, 0.5], index=index),
        split="validation",
    )
    reversed_target_predictions = build_prediction_frame(
        target=pd.Series([0, 1], index=index),
        score=pd.Series([0.5, 0.5], index=index),
        split="validation",
    )

    report = evaluate_predictions(
        predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
    )
    reversed_target_report = evaluate_predictions(
        reversed_target_predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
    )

    assert report.aggregate_metrics["average_precision"] == pytest.approx(0.5)
    assert reversed_target_report.aggregate_metrics["average_precision"] == pytest.approx(0.5)


def test_artifact_writer_emits_reproducible_tables_reports_and_plots(tmp_path) -> None:
    index = _index()
    ticker = index.get_level_values("ticker")
    predictions = build_prediction_frame(
        target=pd.Series(ticker.isin(["CCC.ST", "DDD.ST"]).astype(int), index=index),
        score=pd.Series(
            ticker.map({"AAA.ST": 0.1, "BBB.ST": 0.2, "CCC.ST": 0.8, "DDD.ST": 0.9}),
            index=index,
        ),
        split="validation",
        ranking_return=pd.Series(
            ticker.map({"AAA.ST": -0.01, "BBB.ST": 0.0, "CCC.ST": 0.01, "DDD.ST": 0.02}),
            index=index,
        ),
    )
    config = EvaluationConfig(random_seed=29)
    report = evaluate_predictions(
        predictions,
        features=pd.DataFrame({"feature": 1.0}, index=index),
        config=config,
        ranking_return_column="forward_return_5d",
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    paths = write_evaluation_artifacts(report, first_root)
    write_evaluation_artifacts(report, second_root)

    relative = {path.relative_to(first_root).as_posix() for path in paths}
    assert "summary.json" in relative
    assert "predictions.csv.gz" in relative
    assert "score_quantiles_by_date.csv" in relative
    assert "report.md" in relative
    assert "plots/calibration.svg" in relative
    summary = json.loads((first_root / "summary.json").read_text())
    assert summary["split"] == "validation"
    assert summary["evaluation_config"] == config.to_manifest()
    assert summary["ranking_return_column"] == "forward_return_5d"
    report_markdown = (first_root / "report.md").read_text()
    assert "`forward_return_5d`" in report_markdown
    assert "Do not interpret it as executable strategy P&L" in report_markdown
    assert (first_root / "predictions.csv.gz").read_bytes() == (
        second_root / "predictions.csv.gz"
    ).read_bytes()
