"""Check universal-basis resource formulas and circuit equivalence."""

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator

from qaoa_portfolio.circuit import build_qaoa_circuit
from scripts.analyze_resources import analyze


def test_resource_formulas_and_equivalence() -> None:
    """Dense circuit counts follow the formula and preserve the unitary."""
    n = 4
    fields = np.ones(n) * 0.17
    couplings = np.ones((n, n)) * 0.37
    np.fill_diagonal(couplings, 0)
    for p in [1, 2, 3]:
        gammas, betas = [0.23] * p, [0.19] * p
        result = analyze(n, gammas, betas, fields, couplings)
        pairs = n * (n - 1) // 2
        assert result["unitary_gates"] == n + p * (3 * pairs + 4 * n)
        circuit = build_qaoa_circuit(fields, couplings, gammas, betas, measure=False)
        compiled = transpile(circuit, basis_gates=["h", "rz", "cx"],
                             optimization_level=0, seed_transpiler=0)
        assert Operator(circuit).equiv(Operator(compiled))
