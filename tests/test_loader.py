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

