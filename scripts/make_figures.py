"""Result figure for the project page, from demo/qaoa_demo.json and data/ionq_runs.json.

  docs/figures/qaoa_vs_classical.png   left: share of shots that are valid 5-asset portfolios, by QAOA depth,
                                       on a noiseless local simulation and on IonQ's Forte 1 simulator;
                                       right: gap to the exact optimum for each method's best valid portfolio

Also writes docs/figures/alt_text.md. Colours come from a palette checked for colour-vision
deficiency; series also differ in marker shape and are labelled, so nothing depends on colour alone.

Usage: uv run python scripts/make_figures.py
"""

from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a897f", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
DOT = dict(ms=8.5, mec=SURFACE, mew=2, ls="none")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def main() -> None:
    demo = json.loads((ROOT / "demo/qaoa_demo.json").read_text())
    ionq = json.loads((ROOT / "data/ionq_runs.json").read_text())
    remote = {str(r["depth"]): r for r in ionq["runs"]}
    n, k = demo["n_assets"], demo["cardinality"]
    optimum = demo["exact"]["true_cost"]
    gap = lambda cost: 100 * (cost - optimum) / abs(optimum)
    depths = sorted(demo["depths"], key=int)
    metrics = {p: demo["depths"][p]["metrics"] for p in depths}
    chance = 100 * math.comb(n, k) / 2**n
    shots = metrics[depths[0]]["shots"]
    evals = [demo["depths"][p]["train_evals"] for p in depths]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 5.6), gridspec_kw={"width_ratios": [0.85, 1.3]})
    fig.subplots_adjust(left=0.07, right=0.93, top=0.76, bottom=0.12, wspace=0.8)
    fig.text(0.02, 0.97, "QAOA spends most shots on invalid portfolios and trails every classical baseline",
             fontsize=12.5, fontweight="bold", va="top")
    fig.text(0.02, 0.915, f"Choosing {k} of {n} assets; a valid portfolio holds exactly five, and only valid portfolios are scored. QAOA angles were trained locally\n"
             f"({min(evals)} to {max(evals)} evaluations per depth), then sampled with {shots:,} shots: once on a noiseless local simulation and once on IonQ's Forte 1\n"
             "noise-model simulator through Qollab. One run per setting.",
             fontsize=9, color=INK2, va="top", linespacing=1.35)
    for ax in (a, b):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    a.set_title("Share of shots that are valid portfolios", loc="left", fontsize=9.5, color=INK, pad=8)
    x = np.arange(len(depths))
    local = [100 * metrics[p]["feasible_rate"] for p in depths]
    noisy = [100 * remote[p]["observed_feasible_rate"] for p in depths]
    w = 0.3
    a.bar(x - w / 2 - 0.02, local, width=w, color=BLUE, label="Noiseless local simulation")
    a.bar(x + w / 2 + 0.02, noisy, width=w, color=ORANGE, hatch="//", edgecolor=SURFACE, lw=0, label="IonQ Forte 1 simulator")
    for xi, lo, no in zip(x, local, noisy):
        a.text(xi - w / 2 - 0.02, lo + 1.0, f"{lo:.0f}%", ha="center", fontsize=9, color=INK2)
        a.text(xi + w / 2 + 0.02, no + 1.0, f"{no:.0f}%", ha="center", fontsize=9, color=INK2)
    a.axhline(chance, color=MUTED, lw=1)
    a.text(len(depths) - 0.42, chance, f"random bitstrings:\n{chance:.1f}%", fontsize=8, color=INK2, ha="left", va="center", linespacing=1.3, clip_on=False)
    a.set_xticks(x)
    a.set_xticklabels([f"depth {p}" for p in depths])
    a.set_xlabel("QAOA depth (layers)")
    a.set_ylabel("Valid shots (%)")
    a.set_ylim(0, 44)
    a.set_xlim(-0.6, len(depths) - 0.45)
    a.grid(axis="y", color=GRID, lw=1)
    a.legend(frameon=False, loc="upper right", fontsize=8.5)

    b.set_title("Best valid portfolio found: gap to the exact optimum", loc="left", fontsize=9.5, color=INK, pad=8)
    h = demo["heuristics"]
    rows = [("Rounding the continuous solution", gap(demo["topk_rounded"]["true_cost"]), None),
            (f"Greedy selection ({h['greedy']['objective_evaluations']} evaluations)", gap(h["greedy"]["true_cost"]), None),
            (f"Rounding + swap search ({h['rounded_plus_swaps']['objective_evaluations']} evaluations)", gap(h["rounded_plus_swaps"]["true_cost"]), None),
            (f"{h['random_feasible']['objective_evaluations']:,} random valid portfolios", gap(h["random_feasible"]["true_cost"]), None)]
    rows += [(f"QAOA, depth {p}", gap(metrics[p]["best_sampled_cost"]), p) for p in depths]
    rows.sort(key=lambda r: (r[1], r[2] is not None))
    for i, (_, value, p) in enumerate(rows):
        if p is None:
            b.plot(value, i, color=MUTED, marker="o", **DOT)
            b.text(value + 0.25, i, "optimum" if value < 1e-9 else f"{value:.1f}% worse", va="center", fontsize=9, color=INK2)
            continue
        far = gap(remote[p]["best_feasible_cost"])
        b.plot([value, far], [i, i], color=GRID, lw=2, zorder=1)
        b.plot(value, i, color=BLUE, marker="o", **DOT)
        b.plot(far, i, color=ORANGE, marker="D", **{**DOT, "ms": 7.5})
        b.text(far + 0.25, i, f"{value:.1f}% and {far:.1f}% worse", va="center", fontsize=9, color=INK2)
    b.plot([], [], color=BLUE, marker="o", label="Noiseless local simulation", **DOT)
    b.plot([], [], color=ORANGE, marker="D", label="IonQ Forte 1 simulator", **{**DOT, "ms": 7.5})
    b.plot([], [], color=MUTED, marker="o", label="Classical method", **DOT)
    b.legend(frameon=False, loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 0.98))
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([r[0] for r in rows], color=INK)
    for label, (_, _, p) in zip(b.get_yticklabels(), rows):
        label.set_fontweight("bold" if p else "normal")
    b.set_ylim(len(rows) - 0.5, -0.5)
    b.set_xlim(-0.3, 15)
    b.set_xlabel("Gap to the exact optimum (%, lower is better)")
    b.grid(axis="x", color=GRID, lw=1)
    b.spines["left"].set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "qaoa_vs_classical.png", dpi=200, facecolor=SURFACE)
    alt = ("Two panels. Left, grouped bar chart of the share of QAOA shots that are valid five-asset portfolios, by circuit depth: "
           + "; ".join(f"depth {p}, {lo:.0f}% on a noiseless local simulation and {no:.0f}% on IonQ's Forte 1 simulator" for p, lo, no in zip(depths, local, noisy))
           + f". Random bitstrings would give {chance:.1f}%. Right, dot plot of the gap between each method's best valid portfolio and the exact optimum: "
           + "; ".join(f"{name} {'reaches the optimum' if value < 1e-9 else f'is {value:.1f}% worse'}"
                       + (f" locally and {gap(remote[p]['best_feasible_cost']):.1f}% worse on IonQ's simulator" if p else "") for name, value, p in rows) + ".")
    (OUT / "alt_text.md").write_text("# Alt text for the figures\n\nGenerated by `scripts/make_figures.py`.\n\n**qaoa_vs_classical.png**\n\n" + alt + "\n")
    print("saved", OUT / "qaoa_vs_classical.png", "and alt_text.md")
    for name, value, p in rows:
        print(f"  {value:6.2f}%  {name}" + (f"   IonQ {gap(remote[p]['best_feasible_cost']):.2f}%" if p else ""))


if __name__ == "__main__":
    main()
