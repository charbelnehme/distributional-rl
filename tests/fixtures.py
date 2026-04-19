from __future__ import annotations

import numpy as np
import pandas as pd

from distributional_rl.data import build_feature_dataset


def make_sample_bars(symbol: str = "AAPL", periods: int = 80) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02 14:30:00+00:00", periods=periods, freq="1min")
    base = np.linspace(100.0, 103.0, periods)
    wiggle = 0.15 * np.sin(np.linspace(0, 6, periods))
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


def make_training_frame(periods: int = 80) -> tuple[pd.DataFrame, pd.Series]:
    bars = make_sample_bars(periods=periods)
    return build_feature_dataset(
        bars,
        return_horizon=1,
        volatility_window=5,
        atr_window=5,
        volume_window=5,
    )
