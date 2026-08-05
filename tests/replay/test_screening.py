import pandas as pd

from swingtrader.replay.screening import ScreeningService


def test_continuous_ratio_screening_and_sorting():
    index = pd.MultiIndex.from_product(
        [["yfinance"], ["AAA.ST", "BBB.ST"], pd.date_range("2020-01-01", periods=5)],
        names=["provider", "ticker", "trading_date"],
    )
    prices = pd.DataFrame(index=index, columns=["close"], dtype=float)
    prices.loc[("yfinance", "AAA.ST"), "close"] = [100, 101, 102, 103, 104]
    prices.loc[("yfinance", "BBB.ST"), "close"] = [100, 100, 100, 100, 100]
    configuration = {
        "rules": [
            {
                "expression": {
                    "left": {"kind": "column", "column": "close"},
                    "operation": "divide",
                    "right": {
                        "kind": "indicator",
                        "indicator_id": "sma",
                        "parameters": {"length": 3},
                        "source": "close",
                        "output": "sma",
                    },
                    "lookback_sessions": 1,
                    "aggregation": "latest",
                },
                "comparison": "gt",
                "value": 1.0,
            }
        ],
        "sort": [],
    }
    results = ScreeningService().run(prices, configuration)
    assert [result["ticker"] for result in results] == ["AAA.ST"]
