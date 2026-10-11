"""Solve the classical baselines for the portfolio QUBO instance.

Loads monthly returns, builds the cardinality-constrained QUBO, then runs:

- brute-force global optimum over all ``2^N`` states,
- exact optimum over the ``C(N, k)`` feasible (exactly-``k``) portfolios,
- continuous long-only Markowitz via SLSQP, plus its top-``k`` rounding.

Results are printed as a table and saved to ``data/classical_baseline.json``
— the ground truth the QAOA circuit will be scored against.

Usage: ``.venv/bin/python scripts/bench_classical.py [--k 5] [--q 0.5]``
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from qaoa_portfolio.baseline import (
    best_feasible_exact,
    binary_objective,
    brute_force,
    continuous_markowitz,
    topk_portfolio,
)
from qaoa_portfolio.data import TICKERS, load_returns, mean_and_cov
from qaoa_portfolio.qubo import build_qubo

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "classical_baseline.json"


def run(cardinality: int, risk_aversion: float) -> dict:
    """Run every classical baseline and collect the results.

    Parameters
    ----------
    cardinality:
        Required number of selected assets ``k``.
    risk_aversion:
        Risk-aversion factor ``q``.

    Returns
    -------
    dict
        JSON-serializable results: QUBO parameters, exact optima,
        continuous weights and the top-``k`` rounding with its
        approximation ratio against the feasible optimum.
    """
    returns = load_returns()
    mu, Sigma = mean_and_cov(returns)
    instance = build_qubo(
        mu, Sigma,
        risk_aversion=risk_aversion,
        cardinality=cardinality,
        assets=TICKERS,
    )
    print(f"QUBO: N={len(TICKERS)}, k={cardinality}, q={risk_aversion}, "
          f"A={instance.penalty:.4f}, offset={instance.offset:.4f}")

    print(f"enumerating 2^{len(TICKERS)} = {1 << len(TICKERS)} states ...")
    exact_x, exact_e, n_eval, seconds = brute_force(instance.Q, instance.offset)
    print(f"  done in {seconds:.1f}s")

    feas_x, feas_e = best_feasible_exact(instance.Q, instance.offset, cardinality)
    feas_cost = binary_objective(feas_x, mu, Sigma, risk_aversion)

    weights = continuous_markowitz(mu, Sigma, risk_aversion)
    cont_cost = float(risk_aversion * weights @ np.asarray(Sigma) @ weights
                      - np.asarray(mu) @ weights)
    rounded = topk_portfolio(weights, cardinality)
    rounded_cost = binary_objective(rounded, mu, Sigma, risk_aversion)
    # Costs are return-dominated (negative), so report the gap against the
    # feasible optimum rather than a ratio of negatives.
    relative_gap = (rounded_cost - feas_cost) / abs(feas_cost)

    tickers = np.array(TICKERS)
    print("\nexact penalized optimum : "
          f"E={exact_e:.6f}  |x|={exact_x.sum()}  {list(tickers[exact_x == 1])}")
    print(f"exact feasible optimum  : E={feas_e:.6f}  true cost={feas_cost:.6f}  "
          f"{list(tickers[feas_x == 1])}")
    print(f"continuous Markowitz    : true cost={cont_cost:.6f}")
    print(f"top-{cardinality} rounding        : true cost={rounded_cost:.6f}  "
          f"gap vs feasible opt={relative_gap:+.1%}  {list(tickers[rounded == 1])}")

    return {
        "n_assets": len(TICKERS),
        "assets": TICKERS,
        "cardinality": cardinality,
        "risk_aversion": risk_aversion,
        "penalty": instance.penalty,
        "offset": instance.offset,
        "months": len(returns),
        "exact": {"energy": exact_e, "cardinality": int(exact_x.sum()),
                  "portfolio": [t for t, b in zip(TICKERS, exact_x) if b],
                  "states_evaluated": n_eval, "seconds": seconds},
        "feasible_exact": {"energy": feas_e, "true_cost": feas_cost,
                           "portfolio": [t for t, b in zip(TICKERS, feas_x) if b]},
        "continuous": {"true_cost": cont_cost,
                       "weights": {t: round(float(w), 6) for t, w in zip(TICKERS, weights)}},
        "topk_rounded": {"true_cost": rounded_cost,
                         "relative_gap": relative_gap,
                         "portfolio": [t for t, b in zip(TICKERS, rounded) if b]},
    }


def main() -> None:
    """Parse arguments, run baselines, print the table and save JSON."""
    parser = argparse.ArgumentParser(description="Classical baselines for the portfolio QUBO.")
    parser.add_argument("--k", type=int, default=5, help="cardinality (default: 5)")
    parser.add_argument("--q", type=float, default=0.5, help="risk aversion (default: 0.5)")
    args = parser.parse_args()
    results = run(args.k, args.q)
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
