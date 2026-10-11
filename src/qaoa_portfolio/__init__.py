"""QAOA portfolio optimization package.

Relative imports only: this package is never installed on Qollab, where
``main.py`` reaches it as ``src.qaoa_portfolio`` straight from the repo root.
"""

from .data import TICKERS, load_prices, load_returns, mean_and_cov
from .qubo import QuboInstance, build_qubo

__all__ = ["TICKERS", "QuboInstance", "build_qubo", "load_prices", "load_returns", "mean_and_cov"]
