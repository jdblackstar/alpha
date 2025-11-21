from __future__ import annotations

from typing import Any, Callable, Hashable, Optional

import pandas as pd
from pandas import DataFrame
from pandas.errors import EmptyDataError, ParserError


class DataLoader:
    """Fetch and clean OHLCV data from a CSV file, HTTP URL, or yfinance."""

    @staticmethod
    def load(
        symbol: str,
        filepath: Optional[str] = None,
        url: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> DataFrame:
        """
        Load OHLCV data for one symbol.

        Usage:
            data = DataLoader.load("AAPL", start="2015-01-01")

        The loader tries a local CSV first, then a URL, then yfinance.
        It always returns a DataFrame indexed by datetime with columns
        ['open', 'high', 'low', 'close', 'volume'].
        """
        loaders: list[Callable[[], Optional[DataFrame]]] = []
        if filepath:
            loaders.append(lambda: DataLoader._from_csv(filepath))
        if url:
            loaders.append(lambda: DataLoader._from_url(url))
        loaders.append(lambda: DataLoader._from_yfinance(symbol, start, end))

        last_error: Optional[Exception] = None

        for getter in loaders:
            try:
                frame = getter()
            except ImportError as exc:
                last_error = exc
                continue
            if frame is None:
                continue
            try:
                return DataLoader._clean(frame)
            except ValueError as exc:
                last_error = exc
                continue

        error_message = "DataLoader.load could not fetch data from any provided source."
        if last_error:
            raise ValueError(error_message) from last_error
        raise ValueError(error_message)

    @staticmethod
    def _from_csv(filepath: str) -> Optional[DataFrame]:
        """Read raw OHLCV data from a CSV file."""
        return DataLoader._read_csv_source(filepath)

    @staticmethod
    def _from_url(url: str) -> Optional[DataFrame]:
        """Read raw OHLCV data from an HTTP-accessible CSV file."""
        return DataLoader._read_csv_source(url)

    @staticmethod
    def _from_yfinance(
        symbol: str,
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[DataFrame]:
        """Fetch OHLCV data for one symbol with yfinance."""
        import importlib

        if not symbol:
            return None

        try:
            yf = importlib.import_module("yfinance")
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ImportError(
                "yfinance is required to fetch data via the yfinance fallback."
            ) from exc

        df: DataFrame = yf.download(symbol, start=start, end=end)
        if df.empty:
            return None

        return df.reset_index()

    @staticmethod
    def _read_csv_source(path_or_url: str) -> Optional[DataFrame]:
        """Read a CSV file, returning None when data is missing or unreadable."""
        try:
            frame = pd.read_csv(path_or_url)
        except (FileNotFoundError, OSError, ParserError, EmptyDataError):
            return None
        if frame.empty:
            return None
        return frame

    @staticmethod
    def _clean(df: DataFrame) -> DataFrame:
        """Standardize names, sort by datetime, and drop missing rows."""
        frame = df.copy()
        frame = DataLoader._flatten_columns(frame)
        frame = DataLoader._ensure_datetime_index(frame)
        frame = DataLoader._standardize_columns(frame)
        frame = DataLoader._sort_by_date(frame)
        frame = DataLoader._drop_nas(frame)
        return frame

    @staticmethod
    def _standardize_columns(df: DataFrame) -> DataFrame:
        """Lowercase column names and enforce the OHLCV schema."""
        normalized = df.rename(columns=lambda col: str(col).strip().lower())
        required_columns = ["open", "high", "low", "close", "volume"]

        missing = [col for col in required_columns if col not in normalized.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        return normalized[required_columns]

    @staticmethod
    def _flatten_columns(df: DataFrame) -> DataFrame:
        """
        Reduce MultiIndex columns to their first non-empty level.

        yfinance returns multi-level columns even when requesting a single symbol.
        We only need the field names (open/high/low/close/volume), so drop the rest.
        """
        if isinstance(df.columns, pd.MultiIndex):
            frame = df.copy()
            frame.columns = [
                DataLoader._first_non_empty_label(col) for col in df.columns
            ]
            return frame
        return df

    @staticmethod
    def _first_non_empty_label(label: Hashable | tuple[Any, ...]) -> str:
        if isinstance(label, tuple):
            for entry in label:
                if entry is None:
                    continue
                text = str(entry).strip()
                if text:
                    return text
            return ""
        return str(label)

    @staticmethod
    def _ensure_datetime_index(df: DataFrame) -> DataFrame:
        """Set a datetime index named 'datetime' if needed."""
        frame = df.copy()
        if frame.index.name == "datetime":
            frame.index = pd.to_datetime(frame.index)
            return frame

        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            frame = frame.set_index("datetime")
            return frame

        possible_cols = [col for col in frame.columns if "date" in col.lower()]
        if possible_cols:
            first = possible_cols[0]
            frame["datetime"] = pd.to_datetime(frame[first])
            frame = frame.set_index("datetime")
            return frame

        raise ValueError("Data frame must include a datetime column or index.")

    @staticmethod
    def _sort_by_date(df: DataFrame) -> DataFrame:
        """Sort rows by ascending datetime."""
        return df.sort_index()

    @staticmethod
    def _drop_nas(df: DataFrame) -> DataFrame:
        """Drop rows with missing values."""
        return df.dropna()
