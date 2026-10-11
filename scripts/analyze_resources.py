"""Transpile dense portfolio QAOA to H, RZ and CX for resource accounting.

Run with ``uv run python scripts/analyze_resources.py``. Counts exclude
measurement from the unitary gate total and assume all-to-all connectivity.
This is a universal-basis analysis, not an IonQ native-gate compilation.
"""

import json
from pathlib import Path

import numpy as np
from qiskit import transpile

from qaoa_portfolio.circuit import build_qaoa_circuit

ROOT = Path(__file__).resolve().parents[1]


def analyze(n: int, gammas: list[float], betas: list[float],
            h: np.ndarray, couplings: np.ndarray) -> dict:
    """Measure universal-basis gate counts without optimization or routing.

    Parameters
    ----------
    n : int
        Asset and qubit count.
    gammas, betas : list[float]
        Cost and mixer angles for each layer.
    h, couplings : np.ndarray
        Dense Ising fields and couplings.

    Returns
    -------
    dict
        Qubit count, transpiled depth, gate counts and unitary gate total.
    """
    circuit = build_qaoa_circuit(h, couplings, gammas, betas)
    compiled = transpile(circuit, basis_gates=["h", "rz", "cx"],
                         optimization_level=0, seed_transpiler=0)
    counts = dict(compiled.count_ops())
    p = len(gammas)
    assert counts.get("cx", 0) == p * n * (n - 1)
    assert counts.get("h", 0) == n * (1 + 2 * p)
    assert counts.get("rz", 0) == p * (n * (n - 1) // 2 + 2 * n)
    return {"n_assets": n, "layers": p, "qubits": compiled.num_qubits,
            "depth_with_measurement": compiled.depth(), "counts": counts,
            "unitary_gates": sum(v for g, v in counts.items() if g != "measure")}


def main() -> None:
    """Analyze frozen circuits and synthetic dense instances at several sizes."""
    demo = json.loads((ROOT / "demo/qaoa_demo.json").read_text())
    measured = []
    for entry in demo["depths"].values():
        angles = entry["angles"]
        measured.append(analyze(demo["n_assets"], angles["gammas"], angles["betas"],
                                np.array(demo["h"]), np.array(demo["J"])))
    scaling = []
    for n in [4, 8, 12, 16, 20, 24]:
        coupling = np.ones((n, n)) * 0.37
        np.fill_diagonal(coupling, 0)
        scaling.append(analyze(n, [0.23, 0.41], [0.19, 0.31],
                               np.ones(n) * 0.17, coupling))
    result = {"basis": ["h", "rz", "cx"], "optimization_level": 0,
              "connectivity": "all-to-all; no routing",
              "trained_circuits": measured, "synthetic_p2_scaling": scaling}
    output = ROOT / "data/circuit_resources.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
