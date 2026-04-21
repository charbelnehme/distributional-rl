from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from src.data import AlpacaMarketDataStore, build_feature_dataset
from src.evaluation import summarize_backtest

DistributionalStrategy = None

PENALTY_SCORE = -1e9


@dataclass(frozen=True)
class TrainingDataset:
    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame


@dataclass(frozen=True)
class ValidationConfig:
    n_folds: int = 3
    min_train_rows: int = 20
    min_validation_rows: int = 10


@dataclass(frozen=True)
class OptimizationConfig:
    symbols: tuple[str, ...]
    timeframe: str
    start: datetime
    end: datetime
    storage_root: Path
    download: bool = False
    trials: int = 5
    seed: int = 42
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    artifacts_dir: Path = Path("artifacts/optimization")


@dataclass(frozen=True)
class FoldMetrics:
    fold_index: int
    train_rows: int
    validation_rows: int
    adjusted_sharpe: float | None
    mean_return: float | None
    max_drawdown: float | None
    turnover: float | None
    average_leverage: float | None
    score: float | None
    failed: bool = False
    error: str | None = None


def parse_utc_datetime(raw_value: str) -> datetime:
    value = raw_value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_training_data(
    *,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
    storage_root: Path,
    download: bool,
    return_metadata: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | TrainingDataset:
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

    X, y, metadata = build_feature_dataset(bars, return_metadata=True)
    if X.empty or len(X) < 10:
        raise ValueError("Not enough feature rows were produced to run optimization.")

    if return_metadata:
        return TrainingDataset(features=X, target=y, metadata=metadata)
    return X, y


def _build_walk_forward_folds(
    metadata: pd.DataFrame,
    validation: ValidationConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if validation.n_folds < 1:
        raise ValueError("validation.n_folds must be >= 1.")
    if {"timestamp", "symbol"} - set(metadata.columns):
        raise ValueError("metadata must contain timestamp and symbol columns.")

    ordered = metadata.copy().reset_index().rename(columns={"index": "row_index"})
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    ordered = ordered.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    unique_timestamps = list(pd.Index(ordered["timestamp"]).drop_duplicates())
    if len(unique_timestamps) < validation.n_folds + 1:
        raise ValueError("Not enough unique timestamps to construct walk-forward folds.")

    timestamp_blocks = np.array_split(np.asarray(unique_timestamps, dtype=object), validation.n_folds + 1)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(1, len(timestamp_blocks)):
        train_timestamps = np.concatenate(timestamp_blocks[:fold_index])
        validation_timestamps = np.asarray(timestamp_blocks[fold_index], dtype=object)

        train_mask = ordered["timestamp"].isin(train_timestamps)
        validation_mask = ordered["timestamp"].isin(validation_timestamps)

        train_rows = ordered.loc[train_mask, "row_index"].to_numpy()
        validation_rows = ordered.loc[validation_mask, "row_index"].to_numpy()

        if len(train_rows) < validation.min_train_rows or len(validation_rows) < validation.min_validation_rows:
            continue
        folds.append((train_rows, validation_rows))

    if not folds:
        raise ValueError("No walk-forward folds satisfied the minimum row requirements.")
    return folds


def _build_strategy():
    global DistributionalStrategy
    if DistributionalStrategy is None:
        from src.strategy import DistributionalStrategy as strategy_cls

        DistributionalStrategy = strategy_cls
    return DistributionalStrategy


def _fold_turnover(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 0.0
    deltas = np.abs(np.diff(np.concatenate(([0.0], positions))))
    return float(np.mean(deltas))


def _evaluate_fold(
    *,
    strategy_cls,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: dict[str, object],
    simulation_params: dict[str, object],
    fold_index: int,
) -> FoldMetrics:
    strategy = strategy_cls(
        model_params=model_params,
        simulation_params=simulation_params,
    )
    strategy.fit(X_train, y_train)
    positions = np.asarray(strategy.predict_positions(X_val), dtype=float)
    score_components = summarize_backtest(
        y_val.to_numpy(dtype=float),
        positions,
        transaction_cost=0.0,
    )
    score = float(
        score_components["adjusted_sharpe_ratio"]
        + score_components["mean_return"]
        + 0.5 * score_components["max_drawdown"]
        - 0.05 * score_components["turnover"]
        - 0.01 * score_components["average_leverage"]
    )
    return FoldMetrics(
        fold_index=fold_index,
        train_rows=len(X_train),
        validation_rows=len(X_val),
        adjusted_sharpe=score_components["adjusted_sharpe_ratio"],
        mean_return=score_components["mean_return"],
        max_drawdown=score_components["max_drawdown"],
        turnover=score_components["turnover"],
        average_leverage=score_components["average_leverage"],
        score=score,
    )


def _objective_summary(folds: list[FoldMetrics], failures: int, nan_scores: int) -> dict[str, object]:
    valid_folds = [fold for fold in folds if not fold.failed and fold.score is not None and np.isfinite(fold.score)]
    if not valid_folds:
        return {
            "score": PENALTY_SCORE,
            "fold_count": len(folds),
            "valid_fold_count": 0,
            "failures": failures,
            "nan_scores": nan_scores,
        }

    fold_scores = np.asarray([fold.score for fold in valid_folds], dtype=float)
    fold_sharpes = np.asarray([fold.adjusted_sharpe for fold in valid_folds if fold.adjusted_sharpe is not None], dtype=float)
    fold_returns = np.asarray([fold.mean_return for fold in valid_folds if fold.mean_return is not None], dtype=float)
    score = float(np.median(fold_scores) - 0.1 * np.std(fold_scores))
    if fold_sharpes.size:
        score += 0.05 * float(np.median(fold_sharpes))
    if fold_returns.size:
        score += 0.05 * float(np.mean(fold_returns))
    return {
        "score": score,
        "fold_count": len(folds),
        "valid_fold_count": len(valid_folds),
        "failures": failures,
        "nan_scores": nan_scores,
    }


def make_objective(dataset: TrainingDataset, config: OptimizationConfig):
    validation_folds = _build_walk_forward_folds(dataset.metadata, config.validation)

    def objective(trial: optuna.Trial) -> float:
        strategy_cls = _build_strategy()
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

        fold_metrics: list[FoldMetrics] = []
        failures = 0
        nan_scores = 0

        for fold_index, (train_rows, validation_rows) in enumerate(validation_folds, start=1):
            X_train = dataset.features.iloc[train_rows]
            y_train = dataset.target.iloc[train_rows]
            X_val = dataset.features.iloc[validation_rows]
            y_val = dataset.target.iloc[validation_rows]

            try:
                fold = _evaluate_fold(
                    strategy_cls=strategy_cls,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_params=model_params,
                    simulation_params=simulation_params,
                    fold_index=fold_index,
                )
                if fold.score is None or not np.isfinite(fold.score):
                    nan_scores += 1
                    failures += 1
                    fold = FoldMetrics(
                        **{**asdict(fold), "failed": True, "error": "non-finite score"}
                    )
            except Exception as exc:
                failures += 1
                fold = FoldMetrics(
                    fold_index=fold_index,
                    train_rows=len(X_train),
                    validation_rows=len(X_val),
                    adjusted_sharpe=None,
                    mean_return=None,
                    max_drawdown=None,
                    turnover=None,
                    average_leverage=None,
                    score=None,
                    failed=True,
                    error=str(exc),
                )
            fold_metrics.append(fold)

        summary = _objective_summary(fold_metrics, failures=failures, nan_scores=nan_scores)
        trial.set_user_attr("seed", config.seed)
        trial.set_user_attr("validation_folds", [asdict(fold) for fold in fold_metrics])
        trial.set_user_attr(
            "data_sizes",
            [
                {
                    "fold_index": fold.fold_index,
                    "train_rows": fold.train_rows,
                    "validation_rows": fold.validation_rows,
                }
                for fold in fold_metrics
            ],
        )
        trial.set_user_attr("failures", failures)
        trial.set_user_attr("nan_scores", nan_scores)
        trial.set_user_attr("valid_fold_count", summary["valid_fold_count"])
        trial.set_user_attr("fold_count", summary["fold_count"])
        trial.set_user_attr("aggregate_score", summary["score"])
        return float(summary["score"])

    return objective


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def persist_study_artifacts(
    *,
    study: optuna.Study,
    config: OptimizationConfig,
    dataset: TrainingDataset,
) -> dict[str, Path]:
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    best_trial = study.best_trial
    best_payload = {
        "value": best_trial.value,
        "params": best_trial.params,
        "user_attrs": best_trial.user_attrs,
        "symbols": list(config.symbols),
        "timeframe": config.timeframe,
        "start": config.start,
        "end": config.end,
        "dataset_rows": len(dataset.features),
        "validation": asdict(config.validation),
        "seed": config.seed,
    }

    best_trial_path = config.artifacts_dir / "best_trial.json"
    fold_metrics_path = config.artifacts_dir / "best_trial_folds.json"
    study_summary_path = config.artifacts_dir / "study_summary.json"

    best_trial_path.write_text(json.dumps(best_payload, indent=2, default=_json_default))
    fold_metrics_path.write_text(
        json.dumps(best_trial.user_attrs.get("validation_folds", []), indent=2, default=_json_default)
    )
    study_summary_path.write_text(
        json.dumps(
            {
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "trial_count": len(study.trials),
                "seed": config.seed,
            },
            indent=2,
            default=_json_default,
        )
    )

    return {
        "best_trial": best_trial_path,
        "fold_metrics": fold_metrics_path,
        "study_summary": study_summary_path,
    }


def run_optimization(dataset: TrainingDataset, config: OptimizationConfig) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.set_user_attr("seed", config.seed)
    study.set_user_attr("validation", asdict(config.validation))
    study.set_user_attr("artifacts_dir", str(config.artifacts_dir))
    study.optimize(make_objective(dataset, config), n_trials=config.trials)
    persist_study_artifacts(study=study, config=config, dataset=dataset)
    return study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize strategy hyperparameters.")
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols to use.")
    parser.add_argument("--timeframe", default="1Day", help="Alpaca timeframe, e.g. 1Min or 1Day.")
    parser.add_argument("--start", required=True, help="UTC ISO timestamp, e.g. 2024-01-01T00:00:00+00:00")
    parser.add_argument("--end", required=True, help="UTC ISO timestamp, e.g. 2024-03-01T00:00:00+00:00")
    parser.add_argument("--storage-root", default="data/alpaca", help="Parquet storage root.")
    parser.add_argument("--download", action="store_true", help="Download bars from Alpaca before optimizing.")
    parser.add_argument("--trials", type=int, default=5, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Optuna and model sampling.")
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/optimization",
        help="Directory for best-trial and fold-metric artifacts.",
    )
    parser.add_argument("--validation-folds", type=int, default=3, help="Number of walk-forward validation folds.")
    parser.add_argument(
        "--validation-min-train-rows",
        type=int,
        default=20,
        help="Minimum training rows required for each fold.",
    )
    parser.add_argument(
        "--validation-min-validation-rows",
        type=int,
        default=10,
        help="Minimum validation rows required for each fold.",
    )
    return parser.parse_args()


def _config_from_args(args: argparse.Namespace) -> OptimizationConfig:
    return OptimizationConfig(
        symbols=tuple(args.symbols),
        timeframe=args.timeframe,
        start=parse_utc_datetime(args.start),
        end=parse_utc_datetime(args.end),
        storage_root=Path(args.storage_root),
        download=bool(args.download),
        trials=int(args.trials),
        seed=int(args.seed),
        validation=ValidationConfig(
            n_folds=int(args.validation_folds),
            min_train_rows=int(args.validation_min_train_rows),
            min_validation_rows=int(args.validation_min_validation_rows),
        ),
        artifacts_dir=Path(args.artifacts_dir),
    )


def main() -> None:
    args = parse_args()
    config = _config_from_args(args)

    dataset = load_training_data(
        symbols=list(config.symbols),
        timeframe=config.timeframe,
        start=config.start,
        end=config.end,
        storage_root=config.storage_root,
        download=config.download,
        return_metadata=True,
    )

    study = run_optimization(dataset, config)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"  Artifacts: {config.artifacts_dir}")


if __name__ == "__main__":
    main()
