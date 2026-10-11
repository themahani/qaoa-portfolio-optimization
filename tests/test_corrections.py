"""Regression tests for feasibility, fair objectives and runtime isolation."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from qaoa_portfolio.baseline import binary_objective, continuous_markowitz
from qaoa_portfolio.heuristics import compare_heuristics
from qaoa_portfolio.qubo import build_qubo, qubo_energy


def test_equal_weight_objective_and_relaxation() -> None:
    """Feasible QUBO energies match invested-capital portfolio costs."""
    mu = np.array([0.01, 0.02, 0.03, 0.04])
    cov = np.eye(4) * 0.01
    x = np.array([0, 0, 1, 1])
    instance = build_qubo(mu, cov, cardinality=2)
    assert np.isclose(qubo_energy(x, instance.Q, instance.offset),
                      binary_objective(x, mu, cov, 0.5))
    w = continuous_markowitz(mu, cov, 0.5, cardinality=2)
    assert w.max() <= 0.5 + 1e-8


def test_heuristics_are_feasible() -> None:
    """Every classical heuristic returns exactly k selected assets."""
    results = compare_heuristics(np.arange(5) * 0.01, np.eye(5) * 0.01,
                                0.5, 2, np.array([0, 0, 0, 1, 1]), samples=20)
    for entry in results.values():
        assert len(entry["indices"]) == 2
        assert entry["objective_evaluations"] > 0


def test_demo_import_without_local_libraries() -> None:
    """Block local-only libraries and installed-package imports in a subprocess."""
    root = Path(__file__).resolve().parents[1]
    code = '''
import sys
import qiskit
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pandas', 'scipy', 'qiskit_aer', 'qaoa_portfolio'}:
            raise ImportError('Local-only import: ' + fullname)
sys.meta_path.insert(0, Blocker())
import main
from src.qaoa_portfolio.circuit import build_qaoa_circuit
import numpy as np
assert build_qaoa_circuit(np.zeros(2), np.zeros((2, 2)), [0.1], [0.2]).num_qubits == 2
'''
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)


def test_no_feasible_samples_have_no_cost(monkeypatch) -> None:
    """An infeasible outcome never becomes a reported portfolio improvement."""
    from scripts import train_qaoa

    monkeypatch.setattr(train_qaoa, "state_probabilities",
                        lambda circuit: np.array([1.0, 0.0, 0.0, 0.0]))
    result = train_qaoa.evaluate_depth(
        np.zeros(2), np.zeros((2, 2)), 0.0, np.arange(4, dtype=float),
        np.array([False, True, True, False]), np.array([0, 1, 1, 2]),
        {"gammas": [0.1], "betas": [0.2]}, np.zeros(2), np.eye(2),
        0.5, 1.0, 1, 1, 0)
    assert result["best_sampled_cost"] is None
    assert result["best_sampled_cardinality"] is None
    assert result["best_sampled_portfolio"] == []
    assert result["feasible_rate"] == 0.0
