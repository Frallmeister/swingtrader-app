# Interactive Entry Labeling

The interactive entry-labeling workflow creates one authoritative binary label for each inspected daily candle. It is intended for manually identifying desirable bullish entries before training a classifier on the existing point-in-time feature set.

The reusable implementation lives in `swingtrader.modeling.labeling`. The user-facing workflow is `notebooks/workflows/modeling/05_interactive_entry_labeling.ipynb`. The notebook owns widget callbacks and mutable selections; the module owns window planning, indicator and outcome calculations, Plotly figure construction, and persistence.

## Label Contract

Each saved candle has exactly one row keyed by:

```text
(provider, ticker, trading_date)
```

The remaining columns describe the current authoritative classification:

```text
label_family
label                 # 0 or 1
labeling_session_id
labeled_at
```

`label_family` is metadata rather than part of the key. A later labeling pass can change both the family and the binary value, but it updates the existing candle row instead of creating parallel labels. The first workflow uses `trend_continuation` and binary labels only.

Within a planned window:

- clicked candles are saved as `label = 1`;
- unclicked candles are saved as `label = 0`;
- candles visible only because the user panned outside the planned window remain unlabeled.

Overlapping planned windows load the existing values and upsert the same natural keys. They provide context and allow correction without duplicating observations.

## Planned Windows and Viewport Independence

A session fixes `window_size`, `step_size`, and `forward_horizon`. The step must not exceed the window size, which prevents unlabeled gaps between planned windows. The default configuration uses 80 displayed sessions and advances 60 sessions, giving 20 sessions of overlap.

The planned start and end dates define labeling coverage and navigation. Plotly zooming and panning only change the temporary viewport. Pressing **Next** always saves the planned window and advances by the configured step, even when the user has panned into neighboring data.

A window is generated only when its complete forward-outcome horizon remains on or before the configured `labeling_end_date`. Set that boundary to the inclusive validation end. The workflow can then label train and validation history without displaying locked-test candles or using locked-test outcomes in the heatmap.

## Chart Contents

The chart contains three aligned rows:

1. OHLC candlesticks with EMA 10, EMA 20, EMA 50, retrospective pivot annotations, selected-entry markers, and hover-driven risk guides;
2. green/red volume bars;
3. a forward-outcome heatmap sharing the candle dates on the x-axis.

EMA colors are fixed for visual consistency:

| Indicator | Color |
| --- | --- |
| EMA 10 | Blue (`#1f77b4`) |
| EMA 20 | Orange (`#ff7f0e`) |
| EMA 50 | Purple (`#9467bd`) |

Pivot highs and lows reuse the established annotation appearance. They are retrospective visual aids: a pivot is aligned to its historical candle but requires right-side observations before it can be confirmed. They must not be interpreted as point-in-time predictor values.

Hovering a candle uses its close as an intuitive visual entry price:

```text
stop_distance = ATR × atr_stop_multiple
stop = close - stop_distance
take_profit = close + reward_risk_ratio × stop_distance
```

The ATR stop multiple and reward/risk ratio are adjustable in the notebook. These lines help visual inspection only; executable backtests continue to apply their own next-open entry and gap-adjustment rules.

## Forward-Outcome Heatmap

For each entry candle and each horizon from 1 through `forward_horizon`, the workflow calculates three commission-aware views. With commission rate `c`, entry close `buy`, and future close `sell`:

```text
entry_cost = buy × (1 + c)
exit_proceeds = sell × (1 - c)
net_pnl = exit_proceeds - entry_cost
```

The available modes are:

- **Net return:** `exit_proceeds / entry_cost - 1`;
- **ATR units:** `net_pnl / ATR`;
- **Risk units:** `net_pnl / (ATR × atr_stop_multiple)`.

The default commission rate is `0.0025`, corresponding to 0.25% on entry and exit. Missing future observations and invalid ATR values remain missing. Heatmap colors are centered at zero and use robust symmetric limits so isolated extreme observations do not flatten the useful range.

The heatmap exposes future outcomes deliberately. Labels therefore mean that an entry was desirable in retrospect given both its preceding setup and subsequent swing-trade opportunity. Future outcomes belong to the target-generation process; model features must still remain point-in-time safe.

## Save, Reset, and Resume

The notebook keeps two in-memory states:

- labels loaded when the planned window opened;
- the current working selection.

**Reset labels** restores the loaded state without writing. **Save** upserts the full planned window without moving. **Next** writes the full window and updates session progress in the same database transaction. A failed save therefore cannot advance past unsaved work.

Session rows retain the ordered ticker list, current ticker/window positions, temporal boundary, fixed window configuration, and latest visual risk calibration. Restarting the notebook resumes the most recently updated unfinished session.

## Running the Workflow

Install the notebook environment and ensure bronze daily prices are populated:

```powershell
uv sync --all-extras --dev --group notebook --group docs
uv run --group notebook jupyter lab
```

Open `notebooks/workflows/modeling/05_interactive_entry_labeling.ipynb`, set the ordered ticker universe and validation end date, then run the notebook from the top. Plotly `FigureWidget` interaction requires the notebook group's `anywidget` dependency.
