"""Cardinality-constrained Markowitz QUBO and its Ising mapping.

The optimization problem is::

    min  q * x^T Sigma x / k^2 - mu^T x / k + A * (sum(x) - k)^2
    s.t. x in {0, 1}^N

where ``mu``/``Sigma`` are the mean vector/covariance of monthly returns,
``q`` is the risk-aversion factor, exactly ``k`` assets must be selected,
and ``A`` penalizes violations of the cardinality constraint.

The QUBO uses the upper-triangular convention::

    cost(x) = x^T Q x + offset,   Q_ij = 0 for i > j

so that ``x @ Q @ x`` evaluates the energy directly (using ``xi^2 = xi``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QuboInstance:
    """A cardinality-constrained portfolio QUBO.

    Attributes
    ----------
    Q:
        Upper-triangular QUBO matrix (``Q[i, j] = 0`` for ``i > j``).
    offset:
        Constant energy shift (``A * k^2``), excluded from ``Q``.
    penalty:
        Cardinality penalty strength ``A`` actually used.
    risk_aversion:
        Risk-aversion factor ``q``.
    cardinality:
        Required number of selected assets ``k``.
    assets:
        Ticker per variable, in column order.
    """

    Q: np.ndarray
    offset: float
    penalty: float
    risk_aversion: float
    cardinality: int
    assets: list[str]


def suggest_penalty(
    mu: pd.Series | np.ndarray,
    Sigma: pd.DataFrame | np.ndarray,
    risk_aversion: float,
    margin: float = 2.0,
) -> float:
    """Suggest a penalty strength that provably dominates the objective.

    Any binary state satisfies ``|q x^T Sigma x - mu^T x| <= scale`` with
    ``scale = q * sum|Sigma| + sum|mu|``, so a cardinality violation (which
    costs at least ``A``) can never pay off when ``A > scale``. The returned
    value is ``margin * scale``. Pass moments already scaled for investment
    weights. With the default margin of two, A exceeds the full possible
    objective range (at most twice scale). This is deliberately conservative.

    Parameters
    ----------
    mu:
        Expected returns per asset.
    Sigma:
        Return covariance matrix.
    risk_aversion:
        Risk-aversion factor ``q``.
    margin:
        Safety factor over the provable bound.

    Returns
    -------
    float
        Suggested penalty strength ``A``.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(Sigma, dtype=float)
    if margin <= 1 or risk_aversion < 0:
        raise ValueError("margin must exceed one and risk_aversion must be nonnegative")
    scale = risk_aversion * float(np.abs(sigma_arr).sum()) + float(np.abs(mu_arr).sum())
    return margin * scale


def build_qubo(
    mu: pd.Series | np.ndarray,
    Sigma: pd.DataFrame | np.ndarray,
    risk_aversion: float = 0.5,
    cardinality: int = 5,
    penalty: float | None = None,
    assets: list[str] | None = None,
) -> QuboInstance:
    """Build the cardinality-constrained portfolio QUBO.

    Expands the squared cardinality penalty including all cross terms, using
    ``xi^2 = xi`` and folds the linear part into the diagonal of ``Q``;
    the constant ``A * k^2`` becomes ``offset``.

    Parameters
    ----------
    mu:
        Expected returns per asset (length N).
    Sigma:
        Return covariance matrix (N x N, symmetrized internally).
    risk_aversion:
        Risk-aversion factor ``q`` weighting variance against return.
    cardinality:
        Required number of selected assets ``k``.
    penalty:
        Cardinality penalty ``A``. When omitted, :func:`suggest_penalty`
        supplies a provably dominating value.
    assets:
        Ticker per variable. Defaults to ``asset_0..asset_{N-1}``.

    Returns
    -------
    QuboInstance
        Upper-triangular QUBO matrix, offset and parameters.
    """
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(Sigma, dtype=float)
    if mu_arr.ndim != 1 or sigma_arr.shape != (len(mu_arr), len(mu_arr)):
        raise ValueError("Expected a mean vector and a matching square covariance")
    if not np.isfinite(mu_arr).all() or not np.isfinite(sigma_arr).all():
        raise ValueError("Moments must be finite")
    if risk_aversion < 0:
        raise ValueError("risk_aversion must be nonnegative")
    sigma_arr = 0.5 * (sigma_arr + sigma_arr.T)
    if np.linalg.eigvalsh(sigma_arr).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite")
    n = mu_arr.shape[0]
    if sigma_arr.shape != (n, n):
        raise ValueError(f"Sigma shape {sigma_arr.shape} != ({n}, {n})")
    if not 1 <= cardinality <= n:
        raise ValueError(f"cardinality k={cardinality} out of range for N={n}")
    # Exactly k selected assets each receive 1/k of the invested capital.
    mu_arr = mu_arr / cardinality
    sigma_arr = sigma_arr / cardinality**2
    if penalty is None:
        penalty = suggest_penalty(mu_arr, sigma_arr, risk_aversion)
    if penalty <= 0:
        raise ValueError("penalty must be positive")

    q_mat = np.zeros((n, n))
    q_mat[np.diag_indices(n)] = (
        risk_aversion * sigma_arr.diagonal() - mu_arr + penalty * (1.0 - 2.0 * cardinality)
    )
    iu = np.triu_indices(n, k=1)
    q_mat[iu] = 2.0 * risk_aversion * sigma_arr[iu] + 2.0 * penalty
    return QuboInstance(
        Q=q_mat,
        offset=penalty * cardinality**2,
        penalty=penalty,
        risk_aversion=risk_aversion,
        cardinality=cardinality,
        assets=assets if assets is not None else [f"asset_{i}" for i in range(n)],
    )


def qubo_energy(
    states: np.ndarray, Q: np.ndarray, offset: float = 0.0
) -> np.ndarray | float:
    """Evaluate the QUBO energy of one or more binary states.

    Parameters
    ----------
    states:
        Binary vector(s) of shape ``(N,)`` or ``(M, N)``.
    Q:
        Upper-triangular QUBO matrix.
    offset:
        Constant energy shift.

    Returns
    -------
    float or np.ndarray
        Scalar energy for a single state, or one energy per row.
    """
    x = np.asarray(states, dtype=float)
    energies = np.einsum("...i,ij,...j->...", x, Q, x) + offset
    return energies


def qubo_to_ising(Q: np.ndarray, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Map a QUBO to an Ising Hamiltonian via ``xi = (1 - zi) / 2``.

    Parameters
    ----------
    Q:
        Upper-triangular QUBO matrix.
    offset:
        QUBO constant shift.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(h, J, ising_offset)`` with fields ``h``, pairwise couplings
        ``J`` (symmetric, zero diagonal) and the constant shift, such that
        ``ising_energy(1 - 2x) == qubo_energy(x)``.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    diag = Q.diagonal()
    off_row = Q.sum(axis=1) - diag  # sum_{j>i} Q_ij per row i
    off_col = Q.sum(axis=0) - diag  # sum_{j<i} Q_ji per column i
    h = -diag / 2.0 - (off_row + off_col) / 4.0
    J = np.zeros((n, n))
    iu = np.triu_indices(n, k=1)
    J[iu] = Q[iu] / 4.0
    J = J + J.T
    ising_offset = offset + Q.diagonal().sum() / 2.0 + Q[iu].sum() / 4.0
    return h, J, ising_offset


def ising_energy(
    spins: np.ndarray, h: np.ndarray, J: np.ndarray, offset: float = 0.0
) -> np.ndarray | float:
    """Evaluate the Ising energy of one or more spin configurations.

    Parameters
    ----------
    spins:
        Spin vector(s) with entries in ``{-1, +1}``, shape ``(N,)`` or ``(M, N)``.
    h:
        Local fields, shape ``(N,)``.
    J:
        Symmetric coupling matrix with zero diagonal.
    offset:
        Constant energy shift.

    Returns
    -------
    float or np.ndarray
        Scalar energy for a single configuration, or one per row.
    """
    s = np.asarray(spins, dtype=float)
    energies = np.einsum("...i,i->...", s, h) + np.einsum("...i,ij,...j->...", s, J, s) / 2.0
    return energies + offset
