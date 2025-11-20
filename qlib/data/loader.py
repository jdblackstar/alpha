from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas import DataFrame


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
        frame: Optional[DataFrame] = None

        if filepath:
            frame = DataLoader._from_csv(filepath)
        if frame is None and url:
            frame = DataLoader._from_url(url)
        if frame is None:
            frame = DataLoader._from_yfinance(symbol, start, end)

        if frame is None:
            raise ValueError(
                "DataLoader.load could not fetch data from any provided source."
            )

        return DataLoader._clean(frame)

    @staticmethod
    def _from_csv(filepath: str) -> DataFrame:
        """Read raw OHLCV data from a CSV file."""
        return pd.read_csv(filepath)

    @staticmethod
    def _from_url(url: str) -> DataFrame:
        """Read raw OHLCV data from an HTTP-accessible CSV file."""
        return pd.read_csv(url)

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
                "yfinance is required to pull data when no CSV or URL is provided."
            ) from exc

        df: DataFrame = yf.download(symbol, start=start, end=end)
        if df.empty:
            return None

        return df.reset_index()

    @staticmethod
    def _clean(df: DataFrame) -> DataFrame:
        """Standardize names, sort by datetime, and drop missing rows."""
        frame = df.copy()
        frame = DataLoader._ensure_datetime_index(frame)
        frame = DataLoader._standardize_columns(frame)
        frame = DataLoader._sort_by_date(frame)
        frame = DataLoader._drop_nas(frame)
        return frame

    @staticmethod
    def _standardize_columns(df: DataFrame) -> DataFrame:
        """Lowercase column names and enforce the OHLCV schema."""
        normalized = df.rename(columns=lambda col: col.strip().lower())
        required_columns = ["open", "high", "low", "close", "volume"]

        missing = [col for col in required_columns if col not in normalized.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        return normalized[required_columns]

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
