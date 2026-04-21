from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, Sequence

import numpy as np
import pandas as pd


UTC = timezone.utc
DEFAULT_STORAGE_ROOT = Path("data/alpaca")

_TIMEFRAME_CONFIG: dict[str, tuple[int, str, timedelta]] = {
    "1Min": (1, "Minute", timedelta(days=5)),
    "5Min": (5, "Minute", timedelta(days=30)),
    "15Min": (15, "Minute", timedelta(days=60)),
    "30Min": (30, "Minute", timedelta(days=90)),
    "1Hour": (1, "Hour", timedelta(days=180)),
    "1Day": (1, "Day", timedelta(days=730)),
}

_BAR_COLUMNS = [
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
    "timeframe",
]

_BAR_FILENAME_RE = re.compile(
    r"^bars_(?P<start>\d{8}T\d{6}Z)_(?P<end>\d{8}T\d{6}Z)\.parquet$"
)


class HistoricalBarsClientProtocol(Protocol):
    def get_stock_bars(self, request_params: object) -> object:
        """Return historical bars for the supplied request."""


@dataclass(frozen=True)
class AlpacaCredentials:
    api_key: str
    secret_key: str

    @classmethod
    def from_env(cls) -> "AlpacaCredentials":
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise ValueError(
                "Missing Alpaca credentials. Set both ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY in the environment."
            )
        return cls(api_key=api_key, secret_key=secret_key)


@dataclass(frozen=True)
class FeatureDataset:
    """Feature, target, and metadata bundle for time-series model inputs."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame
    target_name: str = "future_log_return"

    def __post_init__(self) -> None:
        if len(self.features) != len(self.target) or len(self.features) != len(self.metadata):
            raise ValueError("FeatureDataset components must have the same length.")
        if not {"timestamp", "symbol"}.issubset(self.metadata.columns):
            raise ValueError("FeatureDataset metadata must include timestamp and symbol.")

    def __iter__(self) -> Iterator[pd.DataFrame | pd.Series]:
        yield self.features
        yield self.target

    def __len__(self) -> int:
        return len(self.target)

    def subset(self, mask: Sequence[bool] | np.ndarray | pd.Series) -> "FeatureDataset":
        selected = np.asarray(mask, dtype=bool)
        if selected.shape[0] != len(self):
            raise ValueError("Subset mask must match dataset length.")
        return FeatureDataset(
            features=self.features.loc[selected].reset_index(drop=True),
            target=self.target.loc[selected].reset_index(drop=True),
            metadata=self.metadata.loc[selected].reset_index(drop=True),
            target_name=self.target_name,
        )

    def as_tuple(self) -> tuple[pd.DataFrame, pd.Series]:
        return self.features, self.target


def _import_alpaca_market_data() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "src.data requires alpaca-py for Alpaca market data retrieval. "
            "Install project dependencies with `pip install -e .`."
        ) from exc

    return StockHistoricalDataClient, StockBarsRequest, TimeFrame, TimeFrameUnit, (
        DataFeed,
        Adjustment,
    )


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def resolve_timeframe(timeframe: str) -> str:
    timeframe = timeframe.strip()
    if timeframe not in _TIMEFRAME_CONFIG:
        supported = ", ".join(sorted(_TIMEFRAME_CONFIG))
        raise ValueError(
            f"Unsupported timeframe '{timeframe}'. Supported values: {supported}."
        )
    return timeframe


def _alpaca_timeframe(timeframe: str) -> Any:
    _, _, TimeFrame, TimeFrameUnit, _ = _import_alpaca_market_data()
    amount, unit_name, _ = _TIMEFRAME_CONFIG[resolve_timeframe(timeframe)]
    unit = getattr(TimeFrameUnit, unit_name)
    return TimeFrame(amount, unit)


def _timeframe_window(timeframe: str) -> timedelta:
    return _TIMEFRAME_CONFIG[resolve_timeframe(timeframe)][2]


def _iter_request_windows(
    start: datetime,
    end: datetime,
    timeframe: str,
) -> list[tuple[datetime, datetime]]:
    windows: list[tuple[datetime, datetime]] = []
    cursor = _normalize_datetime(start)
    normalized_end = _normalize_datetime(end)
    step = _timeframe_window(timeframe)

    while cursor < normalized_end:
        next_cursor = min(cursor + step, normalized_end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor

    return windows


def _empty_bars_frame(timeframe: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=_BAR_COLUMNS
    ).assign(timeframe=timeframe)


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    normalized = []
    for symbol in symbols:
        value = str(symbol).strip()
        if value and value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one symbol is required.")
    return normalized


def _bars_to_dataframe(response: object, timeframe: str) -> pd.DataFrame:
    if hasattr(response, "df"):
        frame = response.df.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.reset_index()
        else:
            frame = frame.reset_index(drop=False)
    elif hasattr(response, "data"):
        rows: list[dict[str, Any]] = []
        for symbol, bars in getattr(response, "data").items():
            for bar in bars:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": getattr(bar, "timestamp"),
                        "open": getattr(bar, "open"),
                        "high": getattr(bar, "high"),
                        "low": getattr(bar, "low"),
                        "close": getattr(bar, "close"),
                        "volume": getattr(bar, "volume"),
                        "trade_count": getattr(bar, "trade_count", np.nan),
                        "vwap": getattr(bar, "vwap", np.nan),
                    }
                )
        frame = pd.DataFrame.from_records(rows)
    else:
        raise TypeError("Unsupported Alpaca bars response type.")

    if frame.empty:
        return _empty_bars_frame(timeframe)

    rename_map = {}
    if "level_0" in frame.columns and "symbol" not in frame.columns:
        rename_map["level_0"] = "symbol"
    if "level_1" in frame.columns and "timestamp" not in frame.columns:
        rename_map["level_1"] = "timestamp"
    frame = frame.rename(columns=rename_map)

    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Alpaca response is missing required columns: {sorted(missing)}")

    if "trade_count" not in frame.columns:
        frame["trade_count"] = np.nan
    if "vwap" not in frame.columns:
        frame["vwap"] = np.nan

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["timeframe"] = timeframe

    return (
        frame[
            [
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "vwap",
                "timeframe",
            ]
        ]
        .sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "timestamp", "timeframe"])
        .reset_index(drop=True)
    )


def _last_rank_percentile(window: np.ndarray) -> float:
    last_value = window[-1]
    return float(np.mean(window <= last_value))


def _resolve_enum_member(enum_type: Any, value: str) -> Any:
    normalized = value.lower()
    for member_name in dir(enum_type):
        if member_name.startswith("_"):
            continue
        member = getattr(enum_type, member_name)
        member_value = getattr(member, "value", None)
        if member_name.lower() == normalized or str(member_value).lower() == normalized:
            return member
    raise ValueError(f"Unsupported value '{value}' for enum {enum_type.__name__}.")


def _validate_bars_frame(bars: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required_columns.difference(bars.columns)
    if missing:
        raise ValueError(f"Bars data is missing required columns: {sorted(missing)}")

    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    if frame.empty:
        raise ValueError("Bars data is empty after dropping missing required values.")

    price_columns = ["open", "high", "low", "close"]
    if (frame[price_columns] <= 0).any().any():
        raise ValueError("Bars data must contain strictly positive prices.")

    if "timeframe" not in frame.columns:
        frame["timeframe"] = "1Day"
    if "trade_count" not in frame.columns:
        frame["trade_count"] = np.nan
    if "vwap" not in frame.columns:
        frame["vwap"] = np.nan

    return (
        frame[
            [
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "vwap",
                "timeframe",
            ]
        ]
        .sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "timestamp", "timeframe"])
        .reset_index(drop=True)
    )


def _parse_filename_window(path: Path) -> tuple[datetime, datetime] | None:
    match = _BAR_FILENAME_RE.match(path.name)
    if not match:
        return None
    start = datetime.strptime(match.group("start"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    end = datetime.strptime(match.group("end"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return start, end


def _overlaps_window(
    file_window: tuple[datetime, datetime] | None,
    start: datetime | None,
    end: datetime | None,
) -> bool:
    if file_window is None:
        return True
    if start is None and end is None:
        return True
    file_start, file_end = file_window
    start_utc = _normalize_datetime(start) if start is not None else None
    end_utc = _normalize_datetime(end) if end is not None else None
    if start_utc is not None and file_end < start_utc:
        return False
    if end_utc is not None and file_start > end_utc:
        return False
    return True


def _candidate_parquet_paths(
    storage_root: Path,
    *,
    symbols: Optional[Sequence[str]] = None,
    timeframe: Optional[str] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[Path]:
    roots: list[Path] = []
    if symbols:
        for symbol in dict.fromkeys(symbols):
            symbol_root = storage_root / f"symbol={symbol}"
            if timeframe is not None:
                roots.append(symbol_root / f"timeframe={timeframe}")
            else:
                roots.append(symbol_root)
    else:
        roots.append(storage_root)

    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for parquet_path in root.rglob("*.parquet"):
            parts = {
                part.split("=", 1)[0]: part.split("=", 1)[1]
                for part in parquet_path.parts
                if "=" in part
            }
            path_timeframe = parts.get("timeframe")
            if timeframe is not None and path_timeframe != timeframe:
                continue
            if not _overlaps_window(_parse_filename_window(parquet_path), start, end):
                continue
            paths.append(parquet_path)

    return sorted(set(paths))


def build_feature_dataset(
    bars: pd.DataFrame,
    *,
    return_horizon: int = 1,
    volatility_window: int = 20,
    atr_window: int = 14,
    volume_window: int = 20,
) -> FeatureDataset:
    if return_horizon < 1:
        raise ValueError("return_horizon must be >= 1.")
    if volatility_window < 2 or atr_window < 1 or volume_window < 1:
        raise ValueError("Feature windows must be positive and volatility_window >= 2.")

    frame = _validate_bars_frame(bars)
    frame["symbol"] = frame["symbol"].astype(str)
    grouped = frame.groupby("symbol", group_keys=False, sort=False)

    log_close = np.log(frame["close"])
    frame["log_return"] = log_close.groupby(frame["symbol"]).diff()

    intrabar_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["bar_portion"] = ((frame["close"] - frame["low"]) / intrabar_range).clip(
        lower=0.0,
        upper=1.0,
    ).fillna(0.5)

    prev_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    true_range_pct = true_range / prev_close.replace(0, np.nan)

    frame["realised_vol"] = grouped["log_return"].transform(
        lambda values: values.rolling(
            window=volatility_window,
            min_periods=volatility_window,
        ).std(ddof=0)
    )
    frame["_true_range_pct"] = true_range_pct
    frame["atr_pct"] = grouped["_true_range_pct"].transform(
        lambda values: values.rolling(window=atr_window, min_periods=atr_window).mean()
    )
    frame["vol_percentile"] = grouped["volume"].transform(
        lambda values: values.rolling(
            window=volume_window,
            min_periods=volume_window,
        ).apply(_last_rank_percentile, raw=True)
    )
    frame["future_log_return"] = grouped["close"].transform(
        lambda values: np.log(values.shift(-return_horizon) / values)
    )

    feature_columns = [
        "bar_portion",
        "log_return",
        "realised_vol",
        "atr_pct",
        "vol_percentile",
    ]
    dataset_columns = [
        "timestamp",
        "symbol",
        "timeframe",
        *feature_columns,
        "future_log_return",
    ]
    dataset = frame.loc[:, dataset_columns].dropna().reset_index(drop=True)

    features = dataset.loc[:, feature_columns].reset_index(drop=True)
    target = dataset["future_log_return"].rename("future_log_return").reset_index(drop=True)
    metadata = dataset.loc[:, ["timestamp", "symbol", "timeframe"]].reset_index(drop=True)
    return FeatureDataset(features=features, target=target, metadata=metadata)


class AlpacaMarketDataStore:
    """
    Retrieve Alpaca historical bars in API-sized chunks and persist them as
    partitioned parquet files. For this repository, local parquet is the best
    fit for large analytical workloads; Spark or HDFS would be unnecessary
    operational overhead for a single-node research project.
    """

    def __init__(
        self,
        *,
        storage_root: Path | str = DEFAULT_STORAGE_ROOT,
        client: Optional[HistoricalBarsClientProtocol] = None,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        sandbox: bool = False,
        feed: Optional[str] = None,
        adjustment: Optional[str] = None,
    ) -> None:
        self.storage_root = Path(storage_root)
        self.client = client
        self.api_key = api_key
        self.secret_key = secret_key
        self.sandbox = sandbox
        self.feed = feed
        self.adjustment = adjustment

    def _client(self) -> HistoricalBarsClientProtocol:
        if self.client is not None:
            return self.client

        StockHistoricalDataClient, _, _, _, _ = _import_alpaca_market_data()
        credentials = AlpacaCredentials(
            api_key=self.api_key or os.getenv("ALPACA_API_KEY", ""),
            secret_key=self.secret_key or os.getenv("ALPACA_SECRET_KEY", ""),
        )
        if not credentials.api_key or not credentials.secret_key:
            credentials = AlpacaCredentials.from_env()

        self.client = StockHistoricalDataClient(
            api_key=credentials.api_key,
            secret_key=credentials.secret_key,
            sandbox=self.sandbox,
        )
        return self.client

    def download_stock_bars(
        self,
        symbols: Sequence[str],
        *,
        start: datetime,
        end: datetime,
        timeframe: str = "1Day",
        feed: Optional[str] = None,
        adjustment: Optional[str] = None,
        persist: bool = True,
    ) -> pd.DataFrame:
        resolve_timeframe(timeframe)
        normalized_symbols = _normalize_symbols(symbols)

        start_utc = _normalize_datetime(start)
        end_utc = _normalize_datetime(end)
        if end_utc <= start_utc:
            raise ValueError("end must be later than start.")

        client = self._client()
        frames: list[pd.DataFrame] = []
        for window_start, window_end in _iter_request_windows(start_utc, end_utc, timeframe):
            request_kwargs: dict[str, Any] = {
                "symbol_or_symbols": normalized_symbols,
                "start": window_start,
                "end": window_end,
            }
            try:
                _, StockBarsRequest, _, _, enums = _import_alpaca_market_data()
                DataFeed, Adjustment = enums
                request_kwargs["timeframe"] = _alpaca_timeframe(timeframe)

                selected_feed = feed or self.feed
                if selected_feed:
                    request_kwargs["feed"] = _resolve_enum_member(DataFeed, selected_feed)
                selected_adjustment = adjustment or self.adjustment
                if selected_adjustment:
                    request_kwargs["adjustment"] = _resolve_enum_member(
                        Adjustment,
                        selected_adjustment,
                    )

                request_params = StockBarsRequest(**request_kwargs)
            except ModuleNotFoundError:
                if self.client is None:
                    raise
                request_kwargs["timeframe"] = timeframe
                request_params = request_kwargs

            response = client.get_stock_bars(request_params)
            frame = _bars_to_dataframe(response, timeframe)
            if not frame.empty:
                frames.append(frame)

        if not frames:
            result = _empty_bars_frame(timeframe)
        else:
            result = (
                pd.concat(frames, ignore_index=True)
                .sort_values(["symbol", "timestamp"])
                .drop_duplicates(["symbol", "timestamp", "timeframe"])
                .reset_index(drop=True)
            )

        if persist and not result.empty:
            self.persist_bars(result)
        return result

    def persist_bars(self, bars: pd.DataFrame) -> list[Path]:
        if bars.empty:
            return []
        if "timeframe" not in bars.columns:
            raise ValueError("Bars data is missing required column: 'timeframe'")

        frame = _validate_bars_frame(bars)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["year"] = frame["timestamp"].dt.year.astype(str)
        frame["month"] = frame["timestamp"].dt.month.astype(str).str.zfill(2)

        written_files: list[Path] = []
        for (symbol, timeframe, year, month), partition in frame.groupby(
            ["symbol", "timeframe", "year", "month"],
            sort=True,
        ):
            directory = (
                self.storage_root
                / f"symbol={symbol}"
                / f"timeframe={timeframe}"
                / f"year={year}"
                / f"month={month}"
            )
            directory.mkdir(parents=True, exist_ok=True)

            partition = partition.drop(columns=["year", "month"]).reset_index(drop=True)
            existing_paths = sorted(directory.glob("*.parquet"))
            existing_frames = [pd.read_parquet(path) for path in existing_paths if path.exists()]
            merged = pd.concat([partition, *existing_frames], ignore_index=True)
            merged = _validate_bars_frame(merged)

            filename = (
                f"bars_{merged['timestamp'].min():%Y%m%dT%H%M%SZ}_"
                f"{merged['timestamp'].max():%Y%m%dT%H%M%SZ}.parquet"
            )
            path = directory / filename
            merged.to_parquet(path, index=False)
            for existing_path in existing_paths:
                if existing_path != path and existing_path.exists():
                    existing_path.unlink()
            written_files.append(path)

        return written_files

    def load_stock_bars(
        self,
        *,
        symbols: Optional[Sequence[str]] = None,
        timeframe: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if timeframe is not None:
            resolve_timeframe(timeframe)

        paths = _candidate_parquet_paths(
            self.storage_root,
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
        )
        if not paths:
            return _empty_bars_frame(timeframe or "1Day")

        frames: list[pd.DataFrame] = []
        for parquet_path in paths:
            frame = pd.read_parquet(parquet_path)
            frame = _validate_bars_frame(frame)
            frames.append(frame)

        frame = pd.concat(frames, ignore_index=True)
        if start is not None:
            frame = frame[frame["timestamp"] >= _normalize_datetime(start)]
        if end is not None:
            frame = frame[frame["timestamp"] <= _normalize_datetime(end)]

        if frame.empty:
            return _empty_bars_frame(timeframe or "1Day")

        return (
            frame.sort_values(["symbol", "timestamp"])
            .drop_duplicates(["symbol", "timestamp", "timeframe"])
            .reset_index(drop=True)
        )
