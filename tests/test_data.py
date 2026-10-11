"""Smoke tests for the data panel."""

from __future__ import annotations

import pathlib

import pandas as pd

from qaoa_portfolio.data import TICKERS, load_returns

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data"
CAP_BYTES = 2 * 1024 * 1024  # Qollab per-file cap


def test_csvs_exist_and_fit_cap() -> None:
    """CSVs are present and each is under Qollab's 2 MB per-file cap."""
    for name in ("prices_daily.csv", "returns_monthly.csv", "universe.json"):
        path = DATA_DIR / name
        assert path.exists(), f"missing {path}; run scripts/pull_data.py"
        assert path.stat().st_size < CAP_BYTES, f"{name} exceeds 2 MB"


def test_returns_panel_shape_and_columns() -> None:
    """Returns panel covers the 20-asset universe with no gaps."""
    returns = load_returns()
    assert list(returns.columns) == TICKERS
    assert len(returns) >= 100, "expected ~120 months (2015-2024)"
    assert not returns.isna().any().any()


def test_prices_cover_full_window() -> None:
    """Daily prices span roughly ten years ending December 2024."""
    prices = pd.read_csv(DATA_DIR / "prices_daily.csv",
                         parse_dates=["date"], index_col="date")
    assert list(prices.columns) == TICKERS
    assert prices.index.min() < pd.Timestamp("2015-06-01")
    assert prices.index.max() > pd.Timestamp("2024-11-01")
    assert not prices.isna().any().any()
