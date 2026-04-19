from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna

from src.data import AlpacaMarketDataStore, build_feature_dataset
from src.metrics import adjusted_sharpe_ratio
from src.strategy import DistributionalStrategy


def load_training_data(
    *,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    storage_root: Path,
    download: bool,
) -> tuple:
    store = AlpacaMarketDataStore(storage_root=storage_root)

    if download:
        bars = store.download_stock_bars(
            symbols,
            start=start,
            end=end,
            timeframe=timeframe,
            persist=True,
        )
    else:
        bars = store.load_stock_bars(
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )

    if bars.empty:
        raise ValueError(
            "No market data available for optimization. Download bars first or run "
            "with --download."
        )

    X, y = build_feature_dataset(bars)
    if X.empty or len(X) < 10:
        raise ValueError("Not enough feature rows were produced to run optimization.")
    return X, y


def make_objective(X, y):
    def objective(trial):
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        dist_name = trial.suggest_categorical("dist_name", ["Normal", "StudentT"])
        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
        leverage_penalty = trial.suggest_float("leverage_penalty", 0.0, 0.05)
        expected_return_weight = trial.suggest_float("expected_return_weight", 0.5, 2.0)

        model_params = {
            "dist_name": dist_name,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
        }
        simulation_params = {
            "grid_points": np.linspace(0, 2, 11),
            "n_samples": 100,
            "leverage_penalty": leverage_penalty,
            "expected_return_weight": expected_return_weight,
        }

        strategy = DistributionalStrategy(
            model_params=model_params,
            simulation_params=simulation_params,
        )

        try:
            strategy.fit(X_train, y_train)
            positions = strategy.predict_positions(X_val)
            score = adjusted_sharpe_ratio(positions * y_val.to_numpy())
        except Exception as exc:
            print(f"Trial failed: {exc}")
            return -1e9

        return score

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize strategy hyperparameters.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols to use.")
    parser.add_argument("--timeframe", default="1Day", help="Alpaca timeframe, e.g. 1Min or 1Day.")
    parser.add_argument("--start", required=True, help="UTC ISO timestamp, e.g. 2024-01-01T00:00:00+00:00")
    parser.add_argument("--end", required=True, help="UTC ISO timestamp, e.g. 2024-03-01T00:00:00+00:00")
    parser.add_argument("--storage-root", default="data/alpaca", help="Parquet storage root.")
    parser.add_argument("--download", action="store_true", help="Download bars from Alpaca before optimizing.")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end).astimezone(timezone.utc)

    X, y = load_training_data(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start=start,
        end=end,
        storage_root=Path(args.storage_root),
        download=args.download,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(X, y), n_trials=args.trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
