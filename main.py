"""Qollab entry point: sample portfolios with a pre-trained QAOA circuit.

What runs here, on the backend picked in Select QPU: one QAOA circuit at a
frozen depth/angle setting. The measured bitstrings are decoded into asset
portfolios and scored with the unpenalized Markowitz cost.

What does not run here: angle training and the classical baselines. Those
were computed locally (exact statevector + exhaustive enumeration) and are
shipped in demo/qaoa_demo.json. See scripts/train_qaoa.py.

Qollab calls main() after this file runs; do not call it at the bottom.
"""

import json

import numpy as np

from src.qaoa_portfolio.circuit import build_qaoa_circuit, counts_to_portfolios


def _true_cost(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, q: float) -> float:
    """Unpenalized Markowitz cost of one binary portfolio."""
    return float(q * x @ Sigma @ x - mu @ x)


def main(shots: int = 4096, depth: int = 2):
    """Run the frozen QAOA circuit once and score the sampled portfolios."""
    demo = json.load(open("demo/qaoa_demo.json"))
    if str(depth) not in demo["depths"]:
        available = sorted(demo["depths"])
        print(f"Depth p={depth} was not pre-trained; using p={available[-1]}. "
              f"Available: {', '.join(available)}")
        depth = int(available[-1])
    entry = demo["depths"][str(depth)]

    n = demo["n_assets"]
    k = demo["cardinality"]
    q = demo["risk_aversion"]
    mu = np.array(demo["mu"])
    Sigma = np.array(demo["Sigma"])
    h = np.array(demo["h"])
    J = np.array(demo["J"])
    gammas = entry["angles"]["gammas"]
    betas = entry["angles"]["betas"]

    circuit = build_qaoa_circuit(h, J, gammas, betas, measure=True)
    print(f"Backend: {getattr(backend, 'name', type(backend).__name__)}")
    print(f"QAOA p={depth}: {circuit.num_qubits} qubits, depth {circuit.depth()}, "
          f"{circuit.count_ops().get('rzz', 0)} two-qubit gates")
    print(f"Submitting 1 circuit as one job, {shots} shots ...")

    result = backend.run(circuit, shots=shots).result()
    decoded = counts_to_portfolios(result.get_counts(), n)

    scored = []
    for portfolio, count in decoded:
        cost = _true_cost(portfolio, mu, Sigma, q)
        scored.append((cost, int(portfolio.sum()), count, portfolio))
    scored.sort(key=lambda item: item[0])
    feasible = [s for s in scored if s[1] == k]
    feas_rate = sum(s[2] for s in feasible) / shots

    print(f"\nTop sampled portfolios (true Markowitz cost, lower is better):")
    for cost, card, count, portfolio in scored[:5]:
        names = [t for t, b in zip(demo["assets"], portfolio) if b]
        flag = "feasible" if card == k else f"|x|={card}, penalized"
        print(f"  cost {cost:+.6f}  {flag}  {count:5d} shots  {names}")

    best = scored[0]
    best_names = [t for t, b in zip(demo["assets"], best[3]) if b]
    opt = demo["exact"]
    topk = demo["topk_rounded"]
    print(f"\nBest sampled this run : {best[0]:+.6f}  {best_names}")
    print(f"Exact feasible optimum: {opt['true_cost']:+.6f}  {opt['portfolio']}")
    print(f"Classical top-{k} rounding : {topk['true_cost']:+.6f}  {topk['portfolio']}")
    print(f"Feasible-sample rate  : {feas_rate:.1%} of {shots} shots "
          f"(precomputed: {entry['metrics']['p_feasible']:.1%} exact)")
    if best[1] == k and best[0] < topk["true_cost"]:
        print("This run's best feasible sample beats the classical top-k rounding.")
    else:
        print("One sample is a demonstration that the circuit runs, not evidence "
              "about accuracy; see the precomputed metrics for the full comparison.")
    print("\nFull comparison, exact distribution + 4096 seeded shots, computed "
          "locally (not in this run):")
    for p, d in demo["depths"].items():
        m = d["metrics"]
        print(f"  p={p}: expected E={m['expected_energy']:+.4f}  "
              f"P(optimal)={m['p_optimal']:.3f}  "
              f"best of {m['shots']} shots={m['best_sampled_cost']:+.6f}")
    print("\nThis effort is supported by Qollab & IonQ.")
