# distributional-rl

`distributional-rl` is a small research package for pulling historical bars
from Alpaca, persisting them as partitioned parquet files, building modeling
features, fitting an NGBoost-based distributional return model, and converting
predicted return distributions into position sizes.

## Scope of this repository

This checkout contains:

- Alpaca historical market-data retrieval
- parquet-backed local storage for large research datasets
- feature engineering from OHLCV bars
- an NGBoost-backed `DistributionalModel`
- implemented risk metrics
- a distributional position-sizing strategy
- unit tests for the above

It still does **not** contain live trading, streaming, or order book handling.

## Installation

Use Python 3.10+.

```bash
./setup_env.sh
source .venv/bin/activate
```

Set credentials before downloading stock data:

```bash
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
```

## Running tests

```bash
python3 -m unittest discover -s tests -v
python3 -m pytest -q
```

## Example

```python
from datetime import datetime, timezone

from distributional_rl import AlpacaMarketDataStore, DistributionalStrategy, build_feature_dataset

store = AlpacaMarketDataStore()
bars = store.download_stock_bars(
    ["AAPL"],
    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
    end=datetime(2024, 3, 1, tzinfo=timezone.utc),
    timeframe="1Day",
)
X, y = build_feature_dataset(bars)

strategy = DistributionalStrategy(
    model_params={"dist_name": "Normal", "n_estimators": 10},
    simulation_params={"n_samples": 250, "grid_points": [0.0, 0.5, 1.0]},
)
strategy.fit(X, y)
positions = strategy.predict_positions(X.tail(5))
print(positions)
```

## Project layout

- `distributional_rl/data.py`: Alpaca retrieval, parquet storage, and feature engineering
- `distributional_rl/model.py`: NGBoost model wrapper
- `distributional_rl/metrics.py`: risk metric calculation
- `distributional_rl/strategy.py`: position sizing logic
- `tests/`: unit tests
