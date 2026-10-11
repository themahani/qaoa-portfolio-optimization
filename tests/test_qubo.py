"""Tests for the QUBO formulation and classical baselines."""

from __future__ import annotations

import numpy as np

from qaoa_portfolio.baseline import (
    best_feasible_exact,
    binary_objective,
    brute_force,
    continuous_markowitz,
    topk_portfolio,
)
from qaoa_portfolio.qubo import (
    build_qubo,
    ising_energy,
    qubo_energy,
    qubo_to_ising,
    suggest_penalty,
)


def _random_instance(n: int = 8, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.005, 0.01, size=n)
    a = rng.normal(size=(n, n))
    Sigma = a @ a.T / n * 0.01 + np.eye(n) * 0.002
    return mu, Sigma, build_qubo(mu, Sigma, risk_aversion=0.5, cardinality=k)


def test_qubo_matches_direct_objective() -> None:
    """QUBO energy equals the expanded Markowitz-plus-penalty objective."""
    mu, Sigma, inst = _random_instance()
    rng = np.random.default_rng(1)
    for _ in range(20):
        x = rng.integers(0, 2, size=len(mu)).astype(float)
        direct = (0.5 * x @ Sigma @ x / inst.cardinality**2 - mu @ x / inst.cardinality
                  + inst.penalty * (x.sum() - inst.cardinality) ** 2)
        assert np.isclose(float(np.asarray(qubo_energy(x, inst.Q, inst.offset))),
                          direct, rtol=1e-9)


def test_ising_mapping_preserves_energies() -> None:
    """Ising energies match QUBO energies under x = (1 - z) / 2."""
    mu, Sigma, inst = _random_instance()
    h, J, off = qubo_to_ising(inst.Q, inst.offset)
    assert np.allclose(np.diag(J), 0.0)
    assert np.allclose(J, J.T)
    rng = np.random.default_rng(2)
    for _ in range(20):
        x = rng.integers(0, 2, size=len(mu)).astype(float)
        spins = 1.0 - 2.0 * x
        assert np.isclose(float(np.asarray(ising_energy(spins, h, J, off))),
                          float(np.asarray(qubo_energy(x, inst.Q, inst.offset))),
                          rtol=1e-9)


def test_brute_force_finds_global_optimum() -> None:
    """Chunked brute force agrees with naive enumeration on N=8."""
    mu, Sigma, inst = _random_instance()
    n = len(mu)
    best = min(
        (float(np.asarray(qubo_energy(
            np.array([(i >> j) & 1 for j in range(n)], dtype=float),
            inst.Q, inst.offset)))
         for i in range(1 << n))
    )
    x, energy, n_eval, _ = brute_force(inst.Q, inst.offset, block=64)
    assert n_eval == 1 << n
    assert np.isclose(energy, best, rtol=1e-12)
    assert x.sum() == inst.cardinality  # penalty dominates: optimum is feasible


def test_best_feasible_exact() -> None:
    """Feasible enumeration returns an exactly-k portfolio with least energy."""
    _, _, inst = _random_instance()
    x, _ = best_feasible_exact(inst.Q, inst.offset, inst.cardinality)
    assert x.sum() == inst.cardinality


def test_suggest_penalty_dominates() -> None:
    """Suggested penalty exceeds the max possible objective gain."""
    rng = np.random.default_rng(3)
    mu = rng.normal(0.005, 0.01, size=10)
    a = rng.normal(size=(10, 10))
    Sigma = a @ a.T / 10 * 0.01
    q = 0.5
    A = suggest_penalty(mu, Sigma, q)
    worst_gain = q * float(np.abs(Sigma).sum()) + float(np.abs(mu).sum())
    assert A > worst_gain


def test_continuous_markowitz_feasible() -> None:
    """SLSQP weights are long-only and fully invested."""
    mu, Sigma, _ = _random_instance()
    w = continuous_markowitz(mu, Sigma, risk_aversion=0.5)
    assert np.isclose(w.sum(), 1.0, atol=1e-8)
    assert (w >= -1e-9).all()


def test_topk_rounding_selects_k() -> None:
    """Rounding keeps exactly the k largest weights."""
    w = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
    x = topk_portfolio(w, 2)
    assert x.tolist() == [0, 1, 0, 1, 0]


def test_binary_objective_matches_qubo_without_penalty() -> None:
    """Unpenalized objective agrees with the QUBO on feasible states."""
    mu, Sigma, inst = _random_instance()
    x, _ = best_feasible_exact(inst.Q, inst.offset, inst.cardinality)
    assert np.isclose(binary_objective(x, mu, Sigma, 0.5),
                      float(np.asarray(qubo_energy(x, inst.Q, inst.offset))),
                      rtol=1e-9)
