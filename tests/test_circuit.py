"""Tests for the QAOA circuit, training and counts decoding."""

from __future__ import annotations

import numpy as np

from qaoa_portfolio.circuit import (
    all_ising_energies,
    build_qaoa_circuit,
    counts_to_portfolios,
    expected_energy,
    state_probabilities,
    train_qaoa,
)
from qaoa_portfolio.qubo import build_qubo, ising_energy, qubo_to_ising


def test_zero_angles_give_uniform_distribution() -> None:
    """At gamma=beta=0 the circuit is H^N: every outcome equally likely."""
    h = np.array([0.3, -0.1, 0.2])
    J = np.array([[0.0, 0.4, 0.0], [0.4, 0.0, -0.2], [0.0, -0.2, 0.0]])
    probs = state_probabilities(build_qaoa_circuit(h, J, [0.0], [0.0], measure=False))
    assert np.allclose(probs, 1 / 8)


def test_circuit_structure() -> None:
    """One qubit per asset, full measurement, one RZZ per coupled pair."""
    n = 6
    rng = np.random.default_rng(0)
    J = rng.normal(size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    circuit = build_qaoa_circuit(np.zeros(n), J, [0.1, 0.2], [0.3, 0.4])
    assert circuit.num_qubits == n
    assert sum(1 for i in circuit.data if i.operation.name == "measure") == n
    n_pairs = n * (n - 1) // 2  # dense test matrix: every pair coupled
    assert circuit.count_ops().get("rzz", 0) == 2 * n_pairs


def test_all_ising_energies_match_pointwise() -> None:
    """Precomputed table agrees with pointwise Ising energies."""
    rng = np.random.default_rng(1)
    n = 6
    h = rng.normal(size=n)
    J = rng.normal(size=(n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0.0)
    table = all_ising_energies(h, J, offset=0.7)
    for s in [0, 1, 17, 63]:
        spins = 1.0 - 2.0 * np.array([(s >> i) & 1 for i in range(n)])
        assert np.isclose(table[s], float(np.asarray(ising_energy(spins, h, J, 0.7))))


def test_counts_decoding_little_endian() -> None:
    """Leftmost outcome character is the highest qubit (asset N-1)."""
    decoded = counts_to_portfolios({"011": 30, "100": 70}, 3)
    assert [shots for _, shots in decoded] == [70, 30]  # most frequent first
    assert decoded[0][0].tolist() == [0, 0, 1]  # qubit 2 set -> asset 2
    assert decoded[1][0].tolist() == [1, 1, 0]


def test_training_improves_over_uniform() -> None:
    """COBYLA training beats the uniform superposition on a small instance."""
    rng = np.random.default_rng(2)
    n, k = 8, 3
    mu = rng.normal(0.005, 0.01, size=n)
    a = rng.normal(size=(n, n))
    Sigma = a @ a.T / n * 0.01 + np.eye(n) * 0.002
    inst = build_qubo(mu, Sigma, risk_aversion=0.5, cardinality=k)
    h, J, off = qubo_to_ising(inst.Q, inst.offset)
    energies = all_ising_energies(h, J, off)
    uniform = float(np.mean(energies))
    result = train_qaoa(h, J, off, depth=1, restarts=2, maxiter=60, seed=0)
    assert result["expected_energy"] < uniform
    assert len(result["gammas"]) == 1 and len(result["betas"]) == 1
    assert result["evals"] > 0 and result["seconds"] >= 0.0


def test_expected_energy_matches_final_state() -> None:
    """Optimizer target equals the brute-force distribution average."""
    h = np.array([0.3, -0.1])
    J = np.array([[0.0, 0.4], [0.4, 0.0]])
    energies = all_ising_energies(h, J)
    params = np.array([0.7, 1.1])
    probs = state_probabilities(build_qaoa_circuit(h, J, [params[0]], [params[1]],
                                                   measure=False))
    assert np.isclose(expected_energy(params, h, J, energies, 1), probs @ energies)
