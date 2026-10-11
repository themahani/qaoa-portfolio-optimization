"""Exact and continuous classical baselines for the portfolio QUBO.

Two complementary baselines, both evaluated against the same QUBO instance:

- :func:`brute_force`: global optimum of the penalized objective over all
  ``2^N`` states (chunked evaluation, ~seconds for N=20).
- :func:`best_feasible_exact`: global optimum over the ``C(N, k)`` feasible
  (exactly-``k``) portfolios, by direct enumeration.
- :func:`continuous_markowitz` + :func:`topk_portfolio`: capped, long-only,
  fully-invested relaxation solved with SLSQP, rounded to the top-``k``
  assets by weight. This is one heuristic, not the strongest classical method.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .qubo import qubo_energy


def brute_force(
    Q: np.ndarray, offset: float = 0.0, block: int = 1 << 16
) -> tuple[np.ndarray, float, int, float]:
    """Minimize the QUBO over all ``2^N`` binary states by chunked enumeration.

    Parameters
    ----------
    Q:
        Upper-triangular QUBO matrix with ``N <= 24`` variables.
    offset:
        Constant energy shift.
    block:
        States evaluated per chunk (bounds peak memory to
        ``block * N`` floats).

    Returns
    -------
    tuple[np.ndarray, float, int, float]
        ``(best_state, best_energy, n_evaluated, seconds)``.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if block <= 0:
        raise ValueError("block must be positive")
    if n > 24:
        raise ValueError(f"N={n} exceeds the 24-variable brute-force limit")
    total = 1 << n
    best_energy = float("inf")
    best_state = np.zeros(n, dtype=np.int8)
    start = time.perf_counter()
    for lo in range(0, total, block):
        hi = min(lo + block, total)
        idx = np.arange(lo, hi)[:, None]
        chunk = ((idx >> np.arange(n)) & 1).astype(np.float64)
        energies = np.asarray(qubo_energy(chunk, Q, offset))
        j = int(np.argmin(energies))
        if energies[j] < best_energy:
            best_energy = float(energies[j])
            best_state = chunk[j].astype(np.int8)
    seconds = time.perf_counter() - start
    return best_state, best_energy, total, seconds


def best_feasible_exact(
    Q: np.ndarray, offset: float = 0.0, cardinality: int = 5
) -> tuple[np.ndarray, float]:
    """Minimize the QUBO over exactly-``k`` portfolios by direct enumeration.

    Parameters
    ----------
    Q:
        Upper-triangular QUBO matrix.
    offset:
        Constant energy shift.
    cardinality:
        Required number of selected assets ``k``.

    Returns
    -------
    tuple[np.ndarray, float]
        ``(best_state, best_energy)`` restricted to states with ``sum == k``.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    best_energy = float("inf")
    best_state = np.zeros(n, dtype=np.int8)
    for combo in itertools.combinations(range(n), cardinality):
        x = np.zeros(n)
        x[list(combo)] = 1.0
        energy = float(np.asarray(qubo_energy(x, Q, offset)))
        if energy < best_energy:
            best_energy = energy
            best_state = x.astype(np.int8)
    return best_state, best_energy


def binary_objective(
    x: np.ndarray,
    mu: pd.Series | np.ndarray,
    Sigma: pd.DataFrame | np.ndarray,
    risk_aversion: float,
) -> float:
    """Evaluate the unpenalized Markowitz cost of a binary portfolio.

    Parameters
    ----------
    x:
        Binary selection vector.
    mu:
        Expected returns per asset.
    Sigma:
        Return covariance matrix.
    risk_aversion:
        Risk-aversion factor ``q``.

    Returns
    -------
    float
        ``q * w^T Sigma w - mu^T w`` with ``w = x / sum(x)``.
    """
    x = np.asarray(x, dtype=float)
    sigma_arr = np.asarray(Sigma, dtype=float)
    sigma_arr = 0.5 * (sigma_arr + sigma_arr.T)
    k = int(x.sum())
    if k == 0:
        raise ValueError("A portfolio must select at least one asset")
    w = x / k
    return float(risk_aversion * w @ sigma_arr @ w - np.asarray(mu, dtype=float) @ w)


def continuous_markowitz(
    mu: pd.Series | np.ndarray,
    Sigma: pd.DataFrame | np.ndarray,
    risk_aversion: float,
    cardinality: int = 1,
) -> np.ndarray:
    """Solve the long-only, fully-invested continuous Markowitz problem.

    Minimizes ``q * w^T Sigma w - mu^T w`` subject to ``sum(w) = 1`` and
    ``0 <= w <= 1/cardinality`` with SLSQP. For the selection problem this
    is its continuous relaxation, not an unconstrained investment baseline.

    Parameters
    ----------
    mu:
        Expected returns per asset.
    Sigma:
        Return covariance matrix.
    risk_aversion:
        Risk-aversion factor ``q``.

    cardinality:
        Weight cap denominator; pass the binary problem's selected count.

    Returns
    -------
    np.ndarray
        Optimal continuous weights (sum to 1, non-negative).

    Raises
    ------
    RuntimeError
        If the optimizer does not converge.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(Sigma, dtype=float)
    sigma_arr = 0.5 * (sigma_arr + sigma_arr.T)
    n = mu_arr.shape[0]
    result = minimize(
        fun=lambda w: risk_aversion * w @ sigma_arr @ w - mu_arr @ w,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0 / cardinality)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"SLSQP did not converge: {result.message}")
    weights = np.clip(result.x, 0.0, None)
    return weights / weights.sum()


def topk_portfolio(weights: np.ndarray, cardinality: int) -> np.ndarray:
    """Round continuous weights to an exactly-``k`` binary portfolio.

    Parameters
    ----------
    weights:
        Continuous portfolio weights.
    cardinality:
        Number of assets to select ``k``.

    Returns
    -------
    np.ndarray
        Binary vector with ones at the ``k`` largest weights.
    """
    weights = np.asarray(weights, dtype=float)
    if not 1 <= cardinality <= len(weights):
        raise ValueError("cardinality must be between one and the asset count")
    x = np.zeros(weights.shape[0], dtype=np.int8)
    x[np.argsort(weights)[-cardinality:]] = 1
    return x
