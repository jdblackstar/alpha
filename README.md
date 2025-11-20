## Quantitative Finance Research Toolkit

This repo hosts my personal finance research code. Most helpers live in the
`qlib` package, which you can import after installing the project in editable
mode.

### Quick start
- Install [uv](https://docs.astral.sh/uv/) and clone the repo.
- Create the environment and install deps: `uv sync`.
- Make the `qlib` package importable everywhere: `uv pip install -e .`
- Run the test suite anytime with `uv run pytest`.

The editable install step keeps the repo and your environment in sync. Any code
change you make is picked up right away without re-installing.

### Data loader
`qlib.data.loader.DataLoader` fetches OHLCV data from a CSV, a URL, or
`yfinance`, cleans it, and returns a tidy `pandas.DataFrame`. See
`tests/test_loader.py` for example usage.
