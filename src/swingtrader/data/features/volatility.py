"""Volatility feature transformations for daily price histories.

This module builds row-aligned, point-in-time volatility features from ordered
daily price observations. It composes reusable indicators from
:mod:`swingtrader.indicators` into model-facing feature columns, deciding which
source columns each indicator uses, how the results are combined, and what the
final feature columns are named. Calculations are isolated by provider/ticker
groups and leave warm-up periods as missing values until each underlying window
has enough prior observations. The family orchestrator returns a copy of the
input dataframe with the final model feature columns appended and does not mutate
its input.
"""

import pandas as pd

from swingtrader.data.market_frame import (
    validate_market_price_index,
    validate_required_columns,
)
from swingtrader.indicators.volatility import (
    adr,
    atr_percent,
    bollinger_bandwidth,
    bollinger_percent_b,
)


def add_volatility_features(
    data: pd.DataFrame,
    *,
    adr_length: int = 20,
    atr_length: int = 14,
    bollinger_length: int = 20,
    bollinger_num_std: float = 2.0,
) -> pd.DataFrame:
    """Return a copy of data with the default volatility feature set added.

    The input must use the canonical market-price MultiIndex with levels
    ``provider``, ``ticker``, and ``trading_date``, in that exact order, plus
    ``high``, ``low``, ``close``, and ``adjusted_close`` columns. The index must
    be unique and sorted. The returned dataframe preserves the input rows and
    appends the final ``adr_percent``, ``atr_percent``,
    ``bollinger_bandwidth``, and ``bollinger_percent_b`` feature columns.

    ADR is the simple moving average of the daily high-low range over
    ``adr_length`` observations. It is normalized by the current raw close to
    produce ``adr_percent``. ADR measures typical intraday price movement and
    does not account for gaps between the previous close and the current
    session.

    ATR is calculated from raw ``high``, ``low``, and ``close`` because True
    Range combines the intraday range with gaps from the previous close. It is
    normalized by the current raw close to produce ``atr_percent``.

    The Bollinger features are calculated from ``adjusted_close`` so their
    rolling mean and standard deviation are not distorted by split and dividend
    discontinuities in the raw close.

    Raw ADR, True Range, ATR, and the Bollinger bands themselves are expressed
    in the input price units and are not comparable across tickers, so the
    orchestrator only appends the scale-invariant columns. Use the standalone
    indicators in :mod:`swingtrader.indicators` directly when absolute
    price-unit values are required.
    """
    validate_market_price_index(data)
    validate_required_columns(data, required_columns={"high", "low", "close", "adjusted_close"})

    data = data.copy()
    adr_block = adr(data, length=adr_length)
    data["adr_percent"] = adr_block["adr_percent"]

    data["atr_percent"] = atr_percent(data, length=atr_length)

    adjusted_close = data.loc[:, "adjusted_close"]
    data["bollinger_bandwidth"] = bollinger_bandwidth(
        adjusted_close, length=bollinger_length, num_std=bollinger_num_std
    )
    data["bollinger_percent_b"] = bollinger_percent_b(
        adjusted_close, length=bollinger_length, num_std=bollinger_num_std
    )
    return data
