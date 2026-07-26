from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swingtrader.modeling.experiments import ModelSpec
from swingtrader.modeling.training import (
    CONSTANT_PRIOR_MODEL_TYPE,
    LOGISTIC_REGRESSION_MODEL_TYPE,
    RANDOM_RANKING_MODEL_TYPE,
    ConstantPriorClassifier,
    DateMatchedRandomRanker,
    RegularizedLogisticRegression,
    deterministic_random_scores,
    fit_baseline_model,
)


def _index(periods: int = 8) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [
            ("yfinance", ticker, trading_date)
            for trading_date in pd.date_range("2026-01-01", periods=periods // 2)
            for ticker in ("AAA.ST", "BBB.ST")
        ],
        names=("provider", "ticker", "trading_date"),
    )


def test_constant_prior_uses_training_prevalence() -> None:
    index = _index()
    features = pd.DataFrame({"feature": np.arange(len(index))}, index=index)
    target = pd.Series([0, 0, 1, 1, 1, 0, 1, 0], index=index)
    model = fit_baseline_model(
        ModelSpec(name="prior", version="1", model_type=CONSTANT_PRIOR_MODEL_TYPE),
        features=features,
        target=target,
        seed=17,
    )
    assert isinstance(model, ConstantPriorClassifier)
    assert model.prior == pytest.approx(0.5)
    assert model.predict_scores(features).eq(0.5).all()


def test_random_ranking_is_reproducible_and_row_order_independent() -> None:
    index = _index()
    first = deterministic_random_scores(index, seed=17)
    second = deterministic_random_scores(index[::-1], seed=17).reindex(index)
    pd.testing.assert_series_equal(first, second)
    assert not first.equals(deterministic_random_scores(index, seed=18))
    model = fit_baseline_model(
        ModelSpec(name="random", version="1", model_type=RANDOM_RANKING_MODEL_TYPE),
        features=pd.DataFrame({"feature": 1.0}, index=index),
        target=pd.Series([0, 1] * 4, index=index),
        seed=17,
    )
    assert isinstance(model, DateMatchedRandomRanker)


def test_logistic_preprocessing_is_fitted_only_on_training_rows() -> None:
    index = _index(12)
    training_features = pd.DataFrame(
        {
            "signal": [-3.0, -2.0, -1.0, np.nan, 1.0, 2.0, 3.0, 4.0, -4.0, -5.0, 5.0, 6.0],
            "constant": 2.0,
            "all_missing": np.nan,
        },
        index=index,
    )
    target = pd.Series((training_features["signal"].fillna(0.0) > 0).astype(int), index=index)
    model = fit_baseline_model(
        ModelSpec(
            name="logistic",
            version="1",
            model_type=LOGISTIC_REGRESSION_MODEL_TYPE,
            hyperparameters={"regularization_strength": 0.1, "max_iter": 500},
        ),
        features=training_features,
        target=target,
        seed=17,
    )
    assert isinstance(model, RegularizedLogisticRegression)
    training_medians = model.preprocessor.medians
    validation = pd.DataFrame(
        {
            "signal": [10_000.0, np.nan],
            "constant": [2.0, 2.0],
            "all_missing": [np.nan, np.nan],
        },
        index=index[:2],
    )
    scores = model.predict_scores(validation)
    assert model.preprocessor.medians == training_medians
    assert scores.between(0.0, 1.0).all()
    assert scores.iloc[0] > scores.iloc[1]
    manifest = model.to_manifest()
    assert manifest["preprocessing"]["medians"]["signal"] == training_medians[0]
    assert manifest["preprocessing"]["scales"]["constant"] == 1.0
    assert manifest["preprocessing"]["medians"]["all_missing"] == 0.0



def test_baseline_fitting_rejects_duplicate_feature_columns() -> None:
    index = _index()
    features = pd.DataFrame(
        np.ones((len(index), 2)),
        index=index,
        columns=["feature", "feature"],
    )
    target = pd.Series([0, 1] * (len(index) // 2), index=index)

    with pytest.raises(ValueError, match="columns must be unique"):
        fit_baseline_model(
            ModelSpec(name="prior", version="1", model_type=CONSTANT_PRIOR_MODEL_TYPE),
            features=features,
            target=target,
            seed=1,
        )

def test_logistic_rejects_one_class_training_data() -> None:
    index = _index()
    with pytest.raises(ValueError, match="both target classes"):
        fit_baseline_model(
            ModelSpec(name="logistic", version="1", model_type=LOGISTIC_REGRESSION_MODEL_TYPE),
            features=pd.DataFrame({"feature": np.arange(len(index))}, index=index),
            target=pd.Series(1, index=index),
            seed=17,
        )
