"""Pre-train QAOA angles locally and freeze them for the Qollab demo.

Loads the QUBO parameters from ``data/classical_baseline.json``, maps to
Ising, trains depths ``p = 1, 2, 3`` with multi-restart COBYLA against exact
statevector expectations, then evaluates each depth by exact distribution
(probability on the optimum, feasible mass) and by seeded finite-shot
sampling. Everything the demo needs — angles, metrics, references — is saved
to ``demo/qaoa_demo.json``.

Usage: ``.venv/bin/python scripts/train_qaoa.py [--depths 1 2 3]``
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from qaoa_portfolio.baseline import binary_objective
from qaoa_portfolio.circuit import (
    all_ising_energies,
    build_qaoa_circuit,
    state_probabilities,
    train_qaoa,
)
from qaoa_portfolio.data import TICKERS, load_returns, mean_and_cov
from qaoa_portfolio.qubo import build_qubo, qubo_to_ising

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_JSON = ROOT / "data" / "classical_baseline.json"
OUT = ROOT / "demo" / "qaoa_demo.json"
SHOT_EVAL = 4096


def _index_of(portfolio: list[str], assets: list[str]) -> int:
    """Integer index of a portfolio (bit ``i`` = asset ``i`` selected)."""
    return sum(1 << assets.index(t) for t in portfolio)


def evaluate_depth(
    h: np.ndarray,
    J: np.ndarray,
    offset: float,
    energies: np.ndarray,
    feasible_mask: np.ndarray,
    bit_sums: np.ndarray,
    angles: dict,
    mu: np.ndarray,
    Sigma: np.ndarray,
    risk_aversion: float,
    penalty: float,
    cardinality: int,
    opt_index: int,
    seed: int,
) -> dict:
    """Score one trained depth by exact distribution and finite shots.

    Parameters
    ----------
    h, J:
        Ising fields and couplings.
    offset:
        Ising constant shift.
    energies:
        Precomputed penalized energy of every configuration.
    feasible_mask:
        Boolean mask of exactly-``k`` configurations.
    bit_sums:
        Cardinality of every configuration.
    angles:
        ``{"gammas": [...], "betas": [...]}`` from :func:`train_qaoa`.
    mu, Sigma:
        Return mean vector and covariance (true-cost evaluation).
    risk_aversion, penalty, cardinality:
        QUBO parameters.
    opt_index:
        Integer index of the exact optimum portfolio.
    seed:
        RNG seed for the finite-shot sampling.

    Returns
    -------
    dict
        Exact-distribution metrics plus seeded-shot sampling results.
    """
    probs = state_probabilities(
        build_qaoa_circuit(h, J, angles["gammas"], angles["betas"], measure=False)
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(probs), size=SHOT_EVAL, p=probs)
    # Penalized energy minus the cardinality term = unpenalized true cost.
    true_costs = energies[draws] - penalty * (bit_sums[draws] - cardinality) ** 2
    best_draw = draws[int(np.argmin(true_costs))]
    best_bits = np.array([(best_draw >> i) & 1 for i in range(len(h))])
    return {
        "expected_energy": float(probs @ energies),
        "p_optimal": float(probs[opt_index]),
        "p_feasible": float(probs[feasible_mask].sum()),
        "shots": SHOT_EVAL,
        "feasible_rate": float(feasible_mask[draws].mean()),
        "hit_optimal": bool(opt_index in draws),
        "best_sampled_cost": float(true_costs.min()),
        "best_sampled_cardinality": int(best_bits.sum()),
        "best_sampled_portfolio": [t for t, b in zip(TICKERS, best_bits) if b],
        "best_sampled_true_cost_check": binary_objective(best_bits, mu, Sigma, risk_aversion),
    }


def main() -> None:
    """Parse arguments, train every depth, evaluate and save the payload."""
    parser = argparse.ArgumentParser(description="Pre-train QAOA angles.")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    baseline = json.loads(BASELINE_JSON.read_text())
    k = baseline["cardinality"]
    q = baseline["risk_aversion"]
    penalty = baseline["penalty"]
    mu_series, sigma_df = mean_and_cov(load_returns())
    mu, Sigma = np.asarray(mu_series), np.asarray(sigma_df)
    instance = build_qubo(mu_series, sigma_df, risk_aversion=q,
                          cardinality=k, penalty=penalty, assets=TICKERS)
    h, J, ising_offset = qubo_to_ising(instance.Q, instance.offset)

    n = len(TICKERS)
    print(f"Ising: N={n}, k={k}, q={q}, A={penalty:.4f}; "
          f"couplings: all {n * (n - 1) // 2} pairs (dense, penalty-induced)")
    energies = all_ising_energies(h, J, ising_offset)
    all_idx = np.arange(1 << n)
    bit_sums = ((all_idx[:, None] >> np.arange(n)) & 1).sum(axis=1)
    feasible_mask = bit_sums == k
    opt_index = _index_of(baseline["feasible_exact"]["portfolio"], TICKERS)
    assert np.isclose(energies[opt_index], baseline["feasible_exact"]["energy"]), \
        "Ising table disagrees with the classical baseline"

    depths = {}
    for depth in args.depths:
        print(f"\ntraining p={depth} ({args.restarts} restarts x {args.maxiter} iters) ...",
              flush=True)
        result = train_qaoa(h, J, ising_offset, depth=depth,
                            restarts=args.restarts, maxiter=args.maxiter, seed=args.seed)
        circuit = build_qaoa_circuit(h, J, result["gammas"], result["betas"])
        stats = {"qubits": circuit.num_qubits, "depth": circuit.depth(),
                 "rzz_gates": circuit.count_ops().get("rzz", 0)}
        metrics = evaluate_depth(h, J, ising_offset, energies, feasible_mask,
                                 bit_sums, result, mu, Sigma, q, penalty, k,
                                 opt_index, args.seed)
        print(f"  expected E={result['expected_energy']:.6f} "
              f"({result['evals']} evals, {result['seconds']:.0f}s)  "
              f"P(opt)={metrics['p_optimal']:.3f}  P(feasible)={metrics['p_feasible']:.3f}")
        print(f"  {SHOT_EVAL} shots: best true cost={metrics['best_sampled_cost']:.6f} "
              f"(opt={baseline['feasible_exact']['true_cost']:.6f}, "
              f"top-k={baseline['topk_rounded']['true_cost']:.6f}), "
              f"feasible rate={metrics['feasible_rate']:.0%}, "
              f"hit optimal={metrics['hit_optimal']}")
        depths[str(depth)] = {"angles": {"gammas": result["gammas"],
                                         "betas": result["betas"]},
                              "train_seconds": result["seconds"],
                              "circuit": stats, "metrics": metrics}

    payload = {
        "n_assets": n, "assets": TICKERS, "cardinality": k,
        "risk_aversion": q, "penalty": penalty, "ising_offset": ising_offset,
        "mu": [round(float(v), 6) for v in mu],
        "Sigma": [[round(float(v), 8) for v in row] for row in Sigma],
        "h": [round(float(v), 8) for v in h],
        "J": [[round(float(v), 8) for v in row] for row in J],
        "exact": baseline["feasible_exact"], "topk_rounded": baseline["topk_rounded"],
        "depths": depths,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload) + "\n")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
