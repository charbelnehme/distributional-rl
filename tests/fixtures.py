from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import build_feature_dataset


def make_sample_bars(
    symbol: str = "AAPL",
    periods: int = 80,
    *,
    start: str = "2024-01-02 14:30:00+00:00",
    price_level: float = 100.0,
    price_scale: float = 1.0,
    naive_timestamps: bool = False,
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="1min")
    if naive_timestamps:
        timestamps = timestamps.tz_convert(None)
    base = np.linspace(price_level, price_level + 3.0 * price_scale, periods)
    wiggle = 0.15 * price_scale * np.sin(np.linspace(0, 6, periods))
    close = base + wiggle
    open_ = close - 0.05
    high = close + 0.10
    low = close - 0.10
    volume = np.linspace(1_000, 4_000, periods)

    return pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": np.arange(periods) + 10,
            "vwap": close - 0.01,
            "timeframe": "1Min",
        }
    )


def make_multi_symbol_bars(periods: int = 80) -> pd.DataFrame:
    left = make_sample_bars(symbol="AAPL", periods=periods, price_level=100.0, price_scale=1.0)
    right = make_sample_bars(
        symbol="MSFT",
        periods=periods,
        price_level=1_000.0,
        price_scale=10.0,
    )
    return pd.concat([left, right], ignore_index=True)


def make_training_frame(periods: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    bars = make_sample_bars(periods=periods)
    return build_feature_dataset(
        bars,
        return_horizon=1,
        volatility_window=5,
        atr_window=5,
        volume_window=5,
    )
