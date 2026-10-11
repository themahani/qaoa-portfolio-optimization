"""Pull the 20-asset price panel for QAOA portfolio optimization.

Sources (all keyless, free):
  - Yahoo Finance chart API: daily adjusted closes, one call per ticker.

Output (all well under the 2 MB per-file Qollab cap):
  - ``data/prices_daily.csv``: daily adjusted closes, 2015-01-01 to 2024-12-31.
  - ``data/returns_monthly.csv``: month-end simple returns derived from the above.
  - ``data/universe.json``: ticker metadata plus per-asset summary statistics.

The 20 tickers map one-to-one onto 20 qubits in the QAOA circuit, leaving
headroom under Qollab's 24-qubit built-in simulator ceiling.

Usage: ``.venv/bin/python scripts/pull_data.py``
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.parse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
PRICES_CSV = DATA_DIR / "prices_daily.csv"
RETURNS_CSV = DATA_DIR / "returns_monthly.csv"
UNIVERSE_JSON = DATA_DIR / "universe.json"

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) qollab-hackathon-research"}
START = datetime(2015, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)  # period2 is exclusive

# Ticker -> (display name, category). 20 tickers = 20 qubits.
# Mixed equity sectors, bonds, commodities, real estate and crypto give the
# covariance matrix the low/negative off-diagonals that make the cardinality-
# constrained QUBO non-trivial.
TICKERS: dict[str, tuple[str, str]] = {
    "SPY": ("SPDR S&P 500 ETF", "us-equity"),
    "QQQ": ("Invesco QQQ (Nasdaq 100)", "us-equity"),
    "IWM": ("iShares Russell 2000 (small-cap)", "us-equity"),
    "DIA": ("SPDR Dow Jones Industrial Average", "us-equity"),
    "XLK": ("Technology Select Sector SPDR", "sector"),
    "XLF": ("Financial Select Sector SPDR", "sector"),
    "XLV": ("Health Care Select Sector SPDR", "sector"),
    "XLE": ("Energy Select Sector SPDR", "sector"),
    "XLI": ("Industrial Select Sector SPDR", "sector"),
    "XLP": ("Consumer Staples Select Sector SPDR", "sector"),
    "XLU": ("Utilities Select Sector SPDR", "sector"),
    "TLT": ("iShares 20+ Year Treasury Bond", "bonds"),
    "LQD": ("iShares Investment Grade Corporate Bond", "bonds"),
    "HYG": ("iShares High Yield Corporate Bond", "bonds"),
    "GLD": ("SPDR Gold Shares", "commodities"),
    "USO": ("United States Oil Fund", "commodities"),
    "EFA": ("iShares MSCI EAFE (developed ex-US)", "international"),
    "EEM": ("iShares MSCI Emerging Markets", "international"),
    "VNQ": ("Vanguard Real Estate (REIT)", "real-estate"),
    "BTC-USD": ("Bitcoin", "crypto"),
}


def _get(url: str, retries: int = 3) -> requests.Response:
    """GET ``url`` with retries on rate limiting.

    Parameters
    ----------
    url:
        Fully qualified URL to fetch.
    retries:
        Number of attempts before giving up.

    Returns
    -------
    requests.Response
        The successful response (``raise_for_status`` already applied).

    Raises
    ------
    requests.HTTPError
        If every attempt fails.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=UA, timeout=60)
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            last = exc
            status = exc.response.status_code if exc.response is not None else None
            if status not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise last  # pragma: no cover - loop always returns or raises


def fetch_adjusted_close(symbol: str, period1: int, period2: int) -> pd.Series:
    """Fetch the daily adjusted-close series for one symbol.

    Parameters
    ----------
    symbol:
        Yahoo Finance ticker, e.g. ``"SPY"`` or ``"BTC-USD"``.
    period1, period2:
        Unix timestamps bounding the query (``period2`` exclusive).

    Returns
    -------
    pd.Series
        Daily adjusted closes indexed by date (floored to calendar day),
        NaNs dropped. Falls back to raw closes if no adjusted series is
        reported.
    """
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol, safe='')}"
        f"?period1={period1}&period2={period2}&interval=1d&events=div%2Csplit"
    )
    result = _get(url).json()["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None)
    dates = dates.floor("D")
    indicators = result["indicators"]
    adj = indicators.get("adjclose", [{}])[0].get("adjclose")
    raw = indicators["quote"][0]["close"]
    values = adj if adj is not None else raw
    series = pd.Series(values, index=dates, dtype=float).dropna()
    return series.groupby(level=0).last().sort_index().rename(symbol)


def pull_prices(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Pull and align daily adjusted closes for every ticker.

    Parameters
    ----------
    tickers:
        Yahoo Finance tickers to fetch, in column order.
    start, end:
        Timezone-aware bounds for the query (``end`` exclusive).

    Returns
    -------
    pd.DataFrame
        Daily closes, outer-joined on date and forward-filled up to 5 days
        (bridges weekend/holiday gaps between 24/7 crypto and exchange-
        traded funds). Leading rows that are still NaN are dropped.
    """
    period1, period2 = int(start.timestamp()), int(end.timestamp())
    frames = []
    for i, symbol in enumerate(tickers):
        print(f"  [{i + 1}/{len(tickers)}] {symbol} ...", flush=True)
        frames.append(fetch_adjusted_close(symbol, period1, period2))
        time.sleep(0.5)  # be polite to the keyless API
    prices = pd.concat(frames, axis=1, sort=True).sort_index()
    prices = prices.ffill(limit=5).dropna()
    # Drop the partial trailing month (24/7 crypto adds days past the last
    # month-end close of exchange-traded funds).
    prices = prices.loc[(prices.index >= start.replace(tzinfo=None))
                        & (prices.index < end.replace(tzinfo=None))]
    return prices.round(4)


def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily closes to month-end simple returns.

    Parameters
    ----------
    prices:
        Daily adjusted closes with a datetime index.

    Returns
    -------
    pd.DataFrame
        ``P_t / P_{t-1} - 1`` on month-end closes; the first (NaN) month
        is dropped.
    """
    monthly = prices.resample("ME").last()
    return (monthly / monthly.shift(1) - 1.0).dropna().round(8)


def main() -> None:
    """Pull the panel, derive monthly returns and write all three outputs."""
    tickers = list(TICKERS)
    print(f"pulling {len(tickers)} tickers from Yahoo Finance chart API ...")
    prices = pull_prices(tickers, START, END)
    returns = to_monthly_returns(prices)

    prices.to_csv(PRICES_CSV, index_label="date")
    returns.to_csv(RETURNS_CSV, index_label="date")

    ann_mean = 12 * returns.mean()
    ann_vol = float(np.sqrt(12)) * returns.std()
    universe = {
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "trading_days": len(prices),
        "months": len(returns),
        "return_type": "monthly simple adjusted-price returns",
        "summary_window": "2015–2024 descriptive only; training uses 2015–2020",
        "assets": {
            symbol: {
                "name": TICKERS[symbol][0],
                "category": TICKERS[symbol][1],
                "ann_mean": round(float(ann_mean[symbol]), 6),
                "ann_vol": round(float(ann_vol[symbol]), 6),
            }
            for symbol in tickers
        },
    }
    UNIVERSE_JSON.write_text(json.dumps(universe, indent=2) + "\n")

    for path in (PRICES_CSV, RETURNS_CSV, UNIVERSE_JSON):
        print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB, "
              f"Qollab file cap 2048 KB)")
    print(f"\ndaily range: {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"months: {len(returns)}")
    print("\nannualized mean / vol per asset:")
    summary = pd.DataFrame(
        {"ann_mean": ann_mean, "ann_vol": ann_vol,
         "worst_month": returns.min(), "best_month": returns.max()}
    ).round(4)
    print(summary)


if __name__ == "__main__":
    main()
