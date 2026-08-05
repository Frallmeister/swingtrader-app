import numpy as np
import pandas as pd

import swingtrader.indicators as public_indicators
from swingtrader.replay.indicators import calculate_indicator, list_indicator_definitions


def market_prices(length=320):
    index = pd.date_range("2020-01-01", periods=length, freq="B")
    close = pd.Series(100 + np.arange(length) * 0.1, index=index)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adjusted_close": close,
            "volume": 10_000 + np.arange(length),
        },
        index=index,
    )


def test_catalogue_contains_every_public_indicator_and_signature_parameter():
    definitions = {item["id"]: item for item in list_indicator_definitions()}
    assert set(definitions) == set(public_indicators.__all__)
    assert {parameter["name"] for parameter in definitions["macd"]["parameters"]} == {"lengths"}
    assert [output["id"] for output in definitions["macd"]["outputs"]] == [
        "macd",
        "macd_signal",
        "macd_histogram",
    ]


def test_multi_output_indicator_is_calculated_as_one_group():
    result = calculate_indicator(
        market_prices(),
        indicator_id="macd",
        parameters={"lengths": [10, 20, 7]},
        source="close",
    )
    assert list(result.columns) == ["macd", "macd_signal", "macd_histogram"]
    assert result.notna().any().all()


def test_literal_signature_parameters_are_exposed_as_choices():
    definitions = {item["id"]: item for item in list_indicator_definitions()}
    parameters = {item["name"]: item for item in definitions["pivot_points_high_low"]["parameters"]}
    assert parameters["kind"]["kind"] == "choice"
    assert parameters["kind"]["choices"] == ("high_low", "balanced")
