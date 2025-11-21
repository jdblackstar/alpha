from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from qlib.data.loader import DataLoader


def _write_sample_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "Date,Open,High,Low,Close,Volume",
                "2024-01-02,100,110,90,105,1000",
                "2024-01-03,105,115,95,110,1500",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_load_prefers_filepath(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "prices.csv")

    frame = DataLoader.load("AAPL", filepath=str(csv_path))

    assert list(frame.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert frame.index.name == "datetime"
    assert frame.shape == (2, 5)


def test_load_uses_url_when_filepath_absent(tmp_path: Path) -> None:
    csv_path = _write_sample_csv(tmp_path / "remote.csv")
    url = csv_path.as_uri()

    frame = DataLoader.load("AAPL", url=url)

    assert frame.index.name == "datetime"
    assert "open" in frame.columns
    # confirm we parsed the URI source rather than falling back to yfinance
    assert frame.loc["2024-01-02", "close"] == 105


def test_load_falls_back_to_url_when_csv_empty(tmp_path: Path) -> None:
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("Date,Open,High,Low,Close,Volume\n", encoding="utf-8")
    remote_csv = _write_sample_csv(tmp_path / "remote.csv")

    frame = DataLoader.load("AAPL", filepath=str(empty_csv), url=remote_csv.as_uri())

    assert frame.shape == (2, 5)
    assert frame.loc["2024-01-02", "high"] == 110


def test_load_falls_back_to_url_when_csv_invalid(tmp_path: Path) -> None:
    invalid_csv = tmp_path / "invalid.csv"
    invalid_csv.write_text(
        "\n".join(
            [
                "Date,Close",
                "2024-01-02,105",
            ]
        ),
        encoding="utf-8",
    )
    remote_csv = _write_sample_csv(tmp_path / "remote.csv")

    frame = DataLoader.load("AAPL", filepath=str(invalid_csv), url=remote_csv.as_uri())

    assert frame.loc["2024-01-02", "close"] == 105


def test_load_falls_back_to_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module
    captured: dict[str, object] = {}

    def fake_import(name: str):
        if name == "yfinance":

            class DummyYF:
                @staticmethod
                def download(symbol: str, start=None, end=None):
                    captured.update({"symbol": symbol, "start": start, "end": end})
                    idx = pd.date_range("2024-01-05", periods=3, freq="D", name="Date")
                    data = {
                        "Open": [1.0, 2.0, 3.0],
                        "High": [1.5, 2.5, 3.5],
                        "Low": [0.5, 1.5, 2.5],
                        "Close": [1.2, 2.2, 3.2],
                        "Volume": [100, 150, 200],
                    }
                    return pd.DataFrame(data, index=idx)

            return DummyYF
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    frame = DataLoader.load("MSFT", start="2024-01-01", end="2024-02-01")

    assert captured == {
        "symbol": "MSFT",
        "start": "2024-01-01",
        "end": "2024-02-01",
    }
    assert frame.index.name == "datetime"
    assert frame.iloc[-1]["close"] == pytest.approx(3.2)


def test_load_flattens_multiindex_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = importlib.import_module

    def fake_import(name: str):
        if name == "yfinance":

            class DummyYF:
                @staticmethod
                def download(symbol: str, start=None, end=None):
                    idx = pd.date_range("2024-02-01", periods=2, freq="D", name="Date")
                    columns = pd.MultiIndex.from_tuples(
                        [
                            ("Open", symbol),
                            ("High", symbol),
                            ("Low", symbol),
                            ("Close", symbol),
                            ("Adj Close", symbol),
                            ("Volume", symbol),
                        ]
                    )
                    data = [
                        [10.0, 12.0, 9.0, 11.0, 11.5, 1_000],
                        [11.0, 13.0, 10.0, 12.0, 12.5, 1_500],
                    ]
                    return pd.DataFrame(data, index=idx, columns=columns)

            return DummyYF
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    frame = DataLoader.load("AAPL")

    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.iloc[0]["close"] == pytest.approx(11.0)


def test_load_handles_missing_yfinance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_import = importlib.import_module

    def fake_import(name: str):
        if name == "yfinance":
            raise ModuleNotFoundError("yfinance missing")
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    with pytest.raises(ValueError) as excinfo:
        DataLoader.load(
            "AAPL",
            filepath=str(tmp_path / "missing.csv"),
            url=str(tmp_path / "missing-url.csv"),
        )

    assert "could not fetch data" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ImportError)
