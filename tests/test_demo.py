"""Tests for the Qollab demo payload and entry point."""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _payload() -> dict:
    return json.loads((ROOT / "demo" / "qaoa_demo.json").read_text())


def test_demo_payload_complete() -> None:
    """Payload has every field main.py needs, with consistent shapes."""
    demo = _payload()
    n = demo["n_assets"]
    assert len(demo["assets"]) == n == 20
    assert len(demo["mu"]) == n
    assert np.shape(demo["Sigma"]) == (n, n)
    assert np.shape(demo["h"]) == (n,)
    assert np.shape(demo["J"]) == (n, n)
    for depth, entry in demo["depths"].items():
        assert len(entry["angles"]["gammas"]) == int(depth)
        assert len(entry["angles"]["betas"]) == int(depth)
        assert entry["metrics"]["shots"] > 0


def test_demo_circuit_stats_match_rebuild() -> None:
    """Stored circuit stats agree with a locally rebuilt circuit."""
    from src.qaoa_portfolio.circuit import build_qaoa_circuit

    demo = _payload()
    h = np.array(demo["h"])
    J = np.array(demo["J"])
    for depth, entry in demo["depths"].items():
        circuit = build_qaoa_circuit(h, J, entry["angles"]["gammas"],
                                     entry["angles"]["betas"])
        assert circuit.num_qubits == entry["circuit"]["qubits"]
        assert circuit.depth() == entry["circuit"]["depth"]
        assert circuit.count_ops().get("rzz", 0) == entry["circuit"]["rzz_gates"]


def test_main_runs_end_to_end_on_aer(monkeypatch, capsys) -> None:
    """main() submits one job to a local simulator and prints the comparison."""
    import sys

    from qiskit_aer import AerSimulator

    sys.path.insert(0, str(ROOT))
    import main as demo_main

    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(demo_main, "backend", AerSimulator(), raising=False)
    demo_main.main(shots=64, depth=1)
    out = capsys.readouterr().out
    assert "Submitting 1 circuit as one job" in out
    assert "Exact feasible optimum" in out
    assert "This effort is supported by Qollab & IonQ." in out
