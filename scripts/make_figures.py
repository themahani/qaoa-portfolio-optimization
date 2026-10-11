"""Generate result and scaling figures from saved artifacts, without reruns.

Use ``uv run python scripts/make_figures.py`` from the repository root.
Plots use labeled values, distinct markers and a colorblind-friendly palette.
Scaling functions are resource/work models, not measured elapsed times.
"""

import json
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#666666"


def relative_gap(cost: float, optimum: float) -> float:
    """Compute percentage excess cost relative to a nonzero exact optimum.

    Parameters
    ----------
    cost, optimum : float
        Feasible objective values; smaller is better.

    Returns
    -------
    float
        Percentage gap, using the absolute optimum as denominator.
    """
    if optimum == 0:
        raise ValueError("Relative gap is undefined for a zero optimum")
    return 100 * (cost - optimum) / abs(optimum)


def gate_count(n: int, p: int = 2) -> int:
    """Count dense universal-basis unitary gates without optimization.

    Parameters
    ----------
    n, p : int
        Number of assets and QAOA layers.

    Returns
    -------
    int
        H, RZ and CX gate total; measurements excluded.
    """
    return n + p * (3 * n * (n - 1) // 2 + 4 * n)


def save(fig: plt.Figure, name: str, alt: str) -> None:
    """Save a PNG and its plain-language alternative text.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Completed figure.
    name, alt : str
        Output stem and accessible description.
    """
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    (OUT / f"{name}.alt.txt").write_text(alt + "\n")
    plt.close(fig)


def main() -> None:
    """Plot saved optimization results and explicit complexity functions."""
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    baseline = json.loads((ROOT / "data/classical_baseline.json").read_text())
    demo = json.loads((ROOT / "demo/qaoa_demo.json").read_text())
    remote = json.loads((ROOT / "data/ionq_runs.json").read_text())["runs"]
    optimum = baseline["feasible_exact"]["true_cost"]

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    labels = ["Classical\noptimum", "Random feasible\nsampling"]
    costs = [optimum, baseline["heuristics"]["random_feasible"]["true_cost"]]
    colors = [GREEN, GRAY]
    for p in [1, 2, 3]:
        labels += [f"Local ideal\np={p}", f"Forte simulator\np={p}"]
        costs += [demo["depths"][str(p)]["metrics"]["best_sampled_cost"],
                  remote[p - 1]["best_feasible_cost"]]
        colors += [BLUE, ORANGE]
    gaps = [relative_gap(c, optimum) for c in costs]
    bars = ax.bar(np.arange(len(labels)), gaps, color=colors)
    ax.bar_label(bars, labels=[f"{g:.2f}%" for g in gaps], padding=4)
    ax.set_xticks(np.arange(len(labels)), labels, fontsize=9)
    ax.set_ylim(0, max(gaps) * 1.2)
    ax.set_ylabel("Gap to exact feasible optimum (%) — lower is better")
    ax.set_title("Strong classical baselines beat QAOA on this instance")
    ax.text(0.5, -0.25, "20 assets · choose 5 · 4,096 final shots per quantum run · one seed/run per setting",
            transform=ax.transAxes, ha="center", fontsize=9)
    save(fig, "solution_quality", "Percentage gaps to the exact optimum: classical methods 0%, random feasible sampling 0.92%; local QAOA p=1,2,3: 2.22%,1.13%,8.28%; Forte simulator: 4.42%,2.22%,8.99%. Lower is better; all sampled costs are feasible.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    depths = np.array([1, 2, 3])
    ideal = [100 * demo["depths"][str(p)]["metrics"]["p_feasible"] for p in depths]
    local = [100 * demo["depths"][str(p)]["metrics"]["feasible_rate"] for p in depths]
    observed = [100 * r["observed_feasible_rate"] for r in remote]
    for values, color, marker, style, label in [
        (ideal, GRAY, "^", "--", "Ideal exact probability"),
        (local, BLUE, "o", "-", "Local observed shots"),
        (observed, ORANGE, "s", "-", "Forte observed shots")]:
        axes[0].plot(depths, values, color=color, marker=marker, linestyle=style, label=label)
    for p, value in zip(depths, observed):
        axes[0].annotate(f"{value:.1f}%", (p, value), xytext=(5, 7), textcoords="offset points")
    axes[0].set(title="Fewer valid portfolios at greater depth", xlabel="QAOA layers p",
                ylabel="Exactly-five-asset outcomes (%)", ylim=(0, 42), xticks=depths)
    axes[0].legend(fontsize=8)
    seconds = [r["execution_seconds"] for r in remote]
    bars = axes[1].bar(depths, seconds, color=ORANGE, width=0.5)
    axes[1].bar_label(bars, labels=[f"{s} s" for s in seconds], padding=4)
    axes[1].set(title="Remote execution time increases", xlabel="QAOA layers p",
                ylabel="Reported circuit execution time (s)", xticks=depths, ylim=(0, 260))
    fig.suptitle("IonQ Forte 1 noise-model simulator: 20 qubits, 4,096 shots")
    fig.text(0.5, -0.035, "One remote run per depth; timing excludes local training, loading and queueing.", ha="center", fontsize=9)
    save(fig, "feasibility_runtime", "At depths 1,2,3, local observed feasibility is 34.3%,20.3%,3.2% and Forte observed feasibility is 16.3%,5.4%,2.0%. Forte reported execution times are 92,155,224 seconds. These are single runs, not noise-isolation experiments.")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
    n = np.arange(5, 25)
    gates = np.array([gate_count(int(v)) for v in n])
    axes[0].plot(n, n, color=GREEN, marker="o", markevery=4, label="Qubits = N")
    axes[0].plot(n, gates, color=BLUE, label="p=2 unitary gates")
    resources = json.loads((ROOT / "data/circuit_resources.json").read_text())
    measured = [r for r in resources["synthetic_p2_scaling"] if r["n_assets"] >= 5]
    axes[0].scatter([r["n_assets"] for r in measured], [r["unitary_gates"] for r in measured],
                    marker="x", color=ORANGE, label="Transpiled gate counts", zorder=3)
    axes[0].set(yscale="log", xlabel="Assets N", ylabel="Logical resource count (log scale)",
                title="Linear qubits; quadratic gates")
    axes[0].legend(fontsize=8)
    work = [
        (2 * n.astype(float)**2, BLUE, "-", "Quantum gate work: pN²"),
        (2 * n.astype(float)**2 * 2.0**n, ORANGE, "--", "Exact simulation: pN²2ᴺ"),
        (n.astype(float)**2 * np.array([comb(int(v), 5) for v in n]), GREEN, ":", "Feasible enumeration: N²C(N,5)")]
    for values, color, style, label in work:
        axes[1].plot(n, values / values[0], color=color, linestyle=style, label=label)
    axes[1].set(yscale="log", xlabel="Assets N; p=2, k=5 fixed",
                ylabel="Work proxy relative to N=5 (log scale)", title="Modeled growth, not elapsed time")
    axes[1].legend(fontsize=8)
    k = np.arange(0, 21)
    combinations = np.array([comb(20, int(v)) for v in k])
    axes[2].plot(k, combinations, color=GREEN, marker="o", label="Feasible portfolios C(20,k)")
    axes[2].scatter([5], [comb(20, 5)], color=ORANGE, s=65, zorder=3)
    axes[2].annotate("k=5: 15,504", (5, comb(20, 5)), xytext=(0, -30), textcoords="offset points", fontsize=9)
    axes[2].set(yscale="log", xlabel="Selected assets k; N=20 fixed",
                ylabel="Feasible portfolio count (log scale)", title="Cardinality changes search-space size")
    axes[2].legend(fontsize=8)
    fig.suptitle("Scaling with asset universe and portfolio cardinality")
    fig.text(0.5, -0.025, "Work proxies are not equal-cost operations or a speedup claim. Fixed p and training/shot budgets do not guarantee solution quality.", ha="center", fontsize=9)
    save(fig, "complexity_scaling", "Left: N qubits and dense p=2 gate totals growing quadratically, checked by transpilation up to 24 assets. Middle: separately normalized theoretical work functions pN squared, pN squared times 2 to N, and N squared times binomial N choose 5; not runtime measurements. Right: at N=20, feasible portfolio count peaks at k=10; k=5 has 15,504 selections. Fixed k enumeration is polynomial, and no quantum speedup is established.")
    for path in OUT.glob("*.png"):
        if path.stat().st_size >= 2_000_000:
            raise ValueError(f"Figure exceeds Qollab file cap: {path}")
    print(f"Wrote three figures and alt-text files to {OUT}")


if __name__ == "__main__":
    main()
