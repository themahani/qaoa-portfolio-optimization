"""Evaluate frozen training portfolios on held-out 2023–2024 returns."""

import json
from pathlib import Path

import numpy as np

from qaoa_portfolio.data import TICKERS, load_returns

ROOT = Path(__file__).resolve().parents[1]


def performance(returns: np.ndarray, weights: np.ndarray,
                transaction_cost: float = 0.001) -> dict:
    """Backtest monthly target-weight rebalancing with trading costs.

    Parameters
    ----------
    returns : np.ndarray
        Held-out monthly simple asset returns.
    weights : np.ndarray
        Frozen target weights summing to one.
    transaction_cost : float
        Cost per unit absolute weight traded (10 basis points by default).
        Includes initial deployment from cash; no terminal liquidation.

    Returns
    -------
    dict
        Net cumulative return, annualized volatility, maximum drawdown and
        total absolute weight turnover. No hyperparameters are fitted here.
    """
    drifted = np.zeros_like(weights)
    net = []
    turnover = []
    for row in returns:
        traded = float(np.abs(weights - drifted).sum())
        gross = float(weights @ row)
        net.append((1 - transaction_cost * traded) * (1 + gross) - 1)
        turnover.append(traded)
        drifted = weights * (1 + row) / (1 + gross)
    wealth = np.r_[1.0, np.cumprod(1 + np.asarray(net))]
    return {"net_return": float(wealth[-1] - 1),
            "annualized_volatility": float(np.std(net, ddof=1) * np.sqrt(12)),
            "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
            "absolute_weight_turnover": float(sum(turnover))}


def main() -> None:
    """Evaluate classical and QAOA portfolios without refitting on test data."""
    baseline = json.loads((ROOT / "data/classical_baseline.json").read_text())
    demo = json.loads((ROOT / "demo/qaoa_demo.json").read_text())
    candidates = {"exact": baseline["feasible_exact"]["portfolio"],
                  "rounded": baseline["topk_rounded"]["portfolio"]}
    for name, entry in baseline["heuristics"].items():
        candidates[name] = [TICKERS[i] for i in entry["indices"]]
    for depth, entry in demo["depths"].items():
        portfolio = entry["metrics"]["best_sampled_portfolio"]
        if portfolio:
            candidates[f"qaoa_p{depth}"] = portfolio
    test = load_returns().loc["2023-01-01":"2024-12-31"]
    results = {}
    for name, portfolio in candidates.items():
        w = np.array([float(t in portfolio) / len(portfolio) for t in TICKERS])
        results[name] = performance(test.to_numpy(), w)
    output = {"train_end": baseline["training_end"],
              "validation": "2021–2022 reserved; not used to tune this fixed configuration",
              "test": "2023–2024", "cost_per_absolute_weight": 0.001,
              "rebalancing": "monthly; fixed equal-weight selected assets",
              "results": results}
    (ROOT / "data/backtest.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
