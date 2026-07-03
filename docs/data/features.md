# Features

Feature generation is planned, not implemented yet.

## Intended Role

Feature code should transform bronze market and macro data into model-ready records. It should be deterministic and rerunnable.

## Initial Feature Ideas

- daily returns;
- rolling returns;
- moving averages;
- volatility measures;
- volume features;
- opening gap features such as next open versus previous close;
- later macro and market-context joins.

## Design Constraints

- Feature code reads from bronze data, not directly from yfinance.
- Warmup periods should be represented explicitly.
- Features should avoid point-in-time leakage.
- Labels should be generated separately from input features.
- Feature tables should support later train, validation, and test splits.