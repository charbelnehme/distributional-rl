from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.data import AlpacaMarketDataStore, FeatureDataset, build_feature_dataset
from src.metrics import ReturnKind, adjusted_sharpe_ratio
from src.strategy import DistributionalStrategy


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationConfig:
    min_train_timestamps: int = 40
    validation_timestamps: int = 20
    step_timestamps: int = 20
    max_folds: int = 5


@dataclass(frozen=True)
class OptimizationConfig:
    symbols: list[str]
    timeframe: str
    start: datetime
    end: datetime
    storage_root: Path = Path("data/alpaca")
    download: bool = False
    trials: int = 5
    seed: int = 13
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass(frozen=True)
class OptimizationResult:
    study: optuna.study.Study
    dataset: FeatureDataset


def load_training_data(
    *,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    storage_root: Path,
    download: bool,
) -> FeatureDataset:
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

    dataset = build_feature_dataset(bars)
    if len(dataset) < 10:
        raise ValueError("Not enough feature rows were produced to run optimization.")
    return dataset


def _sort_dataset(dataset: FeatureDataset) -> FeatureDataset:
    order = (
        dataset.metadata.assign(timestamp=pd.to_datetime(dataset.metadata["timestamp"], utc=True))
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .index.to_numpy()
    )
    return FeatureDataset(
        features=dataset.features.iloc[order].reset_index(drop=True),
        target=dataset.target.iloc[order].reset_index(drop=True),
        metadata=dataset.metadata.iloc[order].reset_index(drop=True),
        target_name=dataset.target_name,
    )


def _parse_utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _walk_forward_splits(
    dataset: FeatureDataset,
    validation: ValidationConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    metadata = dataset.metadata.copy()
    metadata["timestamp"] = pd.to_datetime(metadata["timestamp"], utc=True)
    timestamps = pd.Index(metadata["timestamp"].drop_duplicates().sort_values())
    if len(timestamps) < validation.min_train_timestamps + validation.validation_timestamps:
        raise ValueError("Not enough unique timestamps for walk-forward validation.")

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    start_index = validation.min_train_timestamps
    while start_index + validation.validation_timestamps <= len(timestamps):
        train_timestamps = timestamps[:start_index]
        validation_timestamps = timestamps[
            start_index : start_index + validation.validation_timestamps
        ]
        train_mask = metadata["timestamp"].isin(train_timestamps).to_numpy()
        validation_mask = metadata["timestamp"].isin(validation_timestamps).to_numpy()
        if train_mask.any() and validation_mask.any():
            folds.append((train_mask, validation_mask))
        if len(folds) >= validation.max_folds:
            break
        start_index += validation.step_timestamps

    if not folds:
        raise ValueError("Unable to construct any walk-forward folds.")

    return folds


def _evaluate_validation_fold(
    strategy: DistributionalStrategy,
    train_dataset: FeatureDataset,
    validation_dataset: FeatureDataset,
    *,
    transaction_cost: float,
) -> float:
    strategy.fit(train_dataset.features, train_dataset.target)
    validation = _sort_dataset(validation_dataset)

    portfolio_returns: list[float] = []
    for symbol, symbol_frame in validation.metadata.groupby("symbol", sort=True):
        symbol_mask = validation.metadata["symbol"] == symbol
        symbol_features = validation.features.loc[symbol_mask].reset_index(drop=True)
        symbol_target = validation.target.loc[symbol_mask].reset_index(drop=True)
        if symbol_features.empty:
            continue

        positions = strategy.predict_positions(symbol_features, initial_position=0.0)
        realised_simple_returns = np.expm1(np.asarray(symbol_target, dtype=float))

        previous_position = 0.0
        for position, realised_return in zip(positions, realised_simple_returns):
            turnover = abs(float(position) - previous_position)
            portfolio_return = (float(position) * float(realised_return)) - (
                turnover * transaction_cost
            )
            if portfolio_return <= -1.0:
                raise ValueError("Validation portfolio return fell below -100%.")
            portfolio_returns.append(portfolio_return)
            previous_position = float(position)

    if len(portfolio_returns) < 2:
        raise ValueError("Validation fold produced too few portfolio returns.")

    return adjusted_sharpe_ratio(np.asarray(portfolio_returns, dtype=float))


def make_objective(dataset: FeatureDataset, validation: ValidationConfig, seed: int):
    sorted_dataset = _sort_dataset(dataset)
    folds = _walk_forward_splits(sorted_dataset, validation)

    def objective(trial: optuna.trial.Trial) -> float:
        trial_seed = seed + trial.number
        rng = np.random.default_rng(trial_seed)

        dist_name = trial.suggest_categorical("dist_name", ["Normal", "StudentT"])
        n_estimators = trial.suggest_int("n_estimators", 25, 200)
        learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
        transaction_cost = trial.suggest_float("transaction_cost", 0.0, 0.002)
        leverage_penalty = trial.suggest_float("leverage_penalty", 0.0, 0.05)
        downside_penalty = trial.suggest_float("downside_penalty", 0.0, 0.1)
        expected_return_weight = trial.suggest_float("expected_return_weight", 0.5, 2.0)
        n_samples = trial.suggest_categorical("n_samples", [64, 128, 256])
        allow_shorting = trial.suggest_categorical("allow_shorting", [False, True])

        grid_points = np.linspace(-2.0 if allow_shorting else 0.0, 2.0, 17)
        model_params = {
            "dist_name": dist_name,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": trial_seed,
        }
        simulation_params = {
            "grid_points": grid_points,
            "n_samples": n_samples,
            "transaction_cost": transaction_cost,
            "leverage_penalty": leverage_penalty,
            "downside_penalty": downside_penalty,
            "expected_return_weight": expected_return_weight,
            "allow_shorting": allow_shorting,
            "random_state": trial_seed,
            "return_kind": ReturnKind.LOG,
        }
        strategy = DistributionalStrategy(
            model_params=model_params,
            simulation_params=simulation_params,
        )

        fold_scores: list[float] = []
        try:
            for fold_index, (train_mask, validation_mask) in enumerate(folds):
                train_dataset = sorted_dataset.subset(train_mask)
                validation_dataset = sorted_dataset.subset(validation_mask)
                score = _evaluate_validation_fold(
                    strategy,
                    train_dataset,
                    validation_dataset,
                    transaction_cost=transaction_cost,
                )
                fold_scores.append(float(score))
                trial.set_user_attr(f"fold_{fold_index}_score", float(score))

            if not fold_scores:
                raise ValueError("No validation folds were scored.")

            score_array = np.asarray(fold_scores, dtype=float)
            objective_value = float(np.mean(score_array) - 0.25 * np.std(score_array, ddof=0))
            trial.set_user_attr("fold_scores", fold_scores)
            trial.set_user_attr("mean_fold_score", float(np.mean(score_array)))
            trial.set_user_attr("std_fold_score", float(np.std(score_array, ddof=0)))
            trial.set_user_attr("trial_seed", trial_seed)
            trial.set_user_attr("grid_points", grid_points.tolist())
            return objective_value
        except optuna.TrialPruned:
            raise
        except ValueError as exc:
            trial.set_user_attr("failure_type", "invalid_trial")
            trial.set_user_attr("failure_message", str(exc))
            LOGGER.info("Pruning trial %s: %s", trial.number, exc)
            raise optuna.TrialPruned(str(exc)) from exc
        except Exception as exc:
            trial.set_user_attr("failure_type", type(exc).__name__)
            trial.set_user_attr("failure_message", str(exc))
            LOGGER.exception("Trial %s failed unexpectedly", trial.number)
            raise

    return objective


def run_optimization(config: OptimizationConfig) -> OptimizationResult:
    dataset = load_training_data(
        symbols=config.symbols,
        timeframe=config.timeframe,
        start=config.start,
        end=config.end,
        storage_root=config.storage_root,
        download=config.download,
    )

    sampler = optuna.samplers.TPESampler(seed=config.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    LOGGER.info(
        "Starting optimization with %s trials over %s rows and %s unique timestamps",
        config.trials,
        len(dataset),
        dataset.metadata["timestamp"].nunique(),
    )
    study.optimize(
        make_objective(dataset, config.validation, config.seed),
        n_trials=config.trials,
        gc_after_trial=True,
    )
    return OptimizationResult(study=study, dataset=dataset)


def parse_args() -> OptimizationConfig:
    parser = argparse.ArgumentParser(description="Optimize strategy hyperparameters.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols to use.")
    parser.add_argument("--timeframe", default="1Day", help="Alpaca timeframe, e.g. 1Min or 1Day.")
    parser.add_argument(
        "--start",
        required=True,
        help="UTC ISO timestamp, e.g. 2024-01-01T00:00:00+00:00",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="UTC ISO timestamp, e.g. 2024-03-01T00:00:00+00:00",
    )
    parser.add_argument("--storage-root", default="data/alpaca", help="Parquet storage root.")
    parser.add_argument("--download", action="store_true", help="Download bars from Alpaca before optimizing.")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed.")
    parser.add_argument(
        "--min-train-timestamps",
        type=int,
        default=40,
        help="Minimum unique timestamps used before the first validation fold.",
    )
    parser.add_argument(
        "--validation-timestamps",
        type=int,
        default=20,
        help="Number of timestamps per validation fold.",
    )
    parser.add_argument(
        "--step-timestamps",
        type=int,
        default=20,
        help="Number of timestamps to advance between walk-forward folds.",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=5,
        help="Maximum number of walk-forward folds to evaluate.",
    )
    args = parser.parse_args()

    return OptimizationConfig(
        symbols=args.symbols,
        timeframe=args.timeframe,
        start=_parse_utc_datetime(args.start),
        end=_parse_utc_datetime(args.end),
        storage_root=Path(args.storage_root),
        download=args.download,
        trials=args.trials,
        seed=args.seed,
        validation=ValidationConfig(
            min_train_timestamps=args.min_train_timestamps,
            validation_timestamps=args.validation_timestamps,
            step_timestamps=args.step_timestamps,
            max_folds=args.max_folds,
        ),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = parse_args()
    result = run_optimization(config)

    print("Best trial:")
    trial = result.study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    if trial.user_attrs:
        print("  Diagnostics:")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
