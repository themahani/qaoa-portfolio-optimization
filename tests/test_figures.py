"""Verify plotted resource formulas and committed figure artifacts."""

import json
from pathlib import Path

import numpy as np

from scripts.make_figures import gate_count, relative_gap

ROOT = Path(__file__).resolve().parents[1]


def test_gate_curve_matches_transpilation() -> None:
    """The plotted gate curve agrees with saved compiler measurements."""
    data = json.loads((ROOT / "data/circuit_resources.json").read_text())
    for result in data["synthetic_p2_scaling"]:
        assert gate_count(result["n_assets"], result["layers"]) == result["unitary_gates"]
    assert np.isclose(relative_gap(-0.028432139602790674, -0.028756035548126585), 1.12636, atol=0.001)


def test_figures_fit_file_cap_and_have_alt_text() -> None:
    """Each figure is a nonempty PNG below the cap, with a text description."""
    for name in ["solution_quality", "feasibility_runtime", "complexity_scaling"]:
        image = ROOT / "docs/figures" / f"{name}.png"
        assert 0 < image.stat().st_size < 2_000_000
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (image.with_suffix(".alt.txt")).read_text().strip()
