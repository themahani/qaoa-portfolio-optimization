"""Load the price/return panels written by ``scripts/pull_data.py``."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / "data"
PRICES_CSV = DATA_DIR / "prices_daily.csv"
RETURNS_CSV = DATA_DIR / "returns_monthly.csv"
UNIVERSE_JSON = DATA_DIR / "universe.json"

#: Universe in column order. One ticker maps to one qubit in the QAOA circuit.
TICKERS: list[str] = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLU",
    "TLT", "LQD", "HYG",
    "GLD", "USO",
    "EFA", "EEM",
    "VNQ",
    "BTC-USD",
]


def load_prices() -> pd.DataFrame:
    """Load daily adjusted closes.

    Returns
    -------
    pd.DataFrame
        Daily closes indexed by date, columns in :data:`TICKERS` order.
    """
    return pd.read_csv(PRICES_CSV, parse_dates=["date"], index_col="date")[TICKERS]


def load_returns() -> pd.DataFrame:
    """Load month-end log returns.

    Returns
    -------
    pd.DataFrame
        Monthly ``log(P_t / P_{t-1})`` indexed by month, columns in
        :data:`TICKERS` order.
    """
    return pd.read_csv(RETURNS_CSV, parse_dates=["date"], index_col="date")[TICKERS]


def load_universe() -> dict:
    """Load ticker metadata and summary statistics.

    Returns
    -------
    dict
        Parsed contents of ``data/universe.json``.
    """
    return json.loads(UNIVERSE_JSON.read_text())


def mean_and_cov(returns: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate the mean vector and covariance matrix of monthly returns.

    Parameters
    ----------
    returns:
        Monthly log returns. Loaded from disk when omitted.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        Mean vector ``mu`` and sample covariance ``Sigma`` over the full
        panel — the two inputs of the Markowitz QUBO.
    """
    if returns is None:
        returns = load_returns()
    return returns.mean(), returns.cov()
