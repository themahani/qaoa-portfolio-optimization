# QAOA Portfolio Optimization

A Qiskit prototype for selecting five equally weighted assets from a
twenty-asset universe. Built for the Qollab × IonQ hackathon's Quantum Impact
theme. We report classical wins and invalid quantum samples explicitly.

## Local setup and reproduction

Python 3.14, managed with uv. Qiskit uses the same range as the team's QRC
project: `>=2.5.2,<3.0.0`; Aer is pinned to `0.17.2`.

```bash
uv sync
uv run python scripts/pull_data.py
uv run python scripts/bench_classical.py
uv run python scripts/train_qaoa.py --depths 1 2 3 --restarts 3 --maxiter 200
uv run python scripts/backtest.py
uv run pytest
```

The Yahoo snapshot can change on refetch. Seeded random starts and sampling
aid reproduction; floating-point/platform differences can affect optimization.

## Investment model and data

Select exactly k=5 assets, each with weight 1/k. Minimize
`q * x.T @ covariance @ x / k**2 - mean @ x / k` with q=0.5.
The QUBO adds `A * (sum(x)-k)**2`. Moments use monthly **simple** adjusted-price
returns, not weighted log returns.

The universe is SPY, QQQ, IWM, DIA, XLK, XLF, XLV, XLE, XLI, XLP, XLU,
TLT, LQD, HYG, GLD, USO, EFA, EEM, VNQ and BTC-USD. Treasury and gold
correlations are not assumed negative. USO tracks oil futures, not spot oil.
BTC trades on weekends; the daily panel carries ETF closes across those
dates. Monthly returns use each month's last available close.

Files are individually below 2,000,000 bytes. Price data covers 2015–2024.
Train moments on 2015–2020; reserve 2021–2022; backtest only on 2023–2024.
No configuration is currently selected using the reserved interval.
The universe and experiment were designed after inspecting historical data;
this split is an exploratory retrospective test, not a preregistered study.

## Comparisons and limitations

`data/classical_baseline.json` records exhaustive and feasible-only optima,
the capped continuous relaxation, rounding, random feasible sampling, greedy
selection and swap-search baselines. The relaxation has weights summing to
one and bounded above by 1/k, matching the binary investment budget.

On the corrected training instance, rounding already attains the exact
optimum (approximately -0.028756). The earlier claim of beating rounding
used inconsistent capital scaling and included infeasible samples. It is
withdrawn. Exact enumeration is cheap at twenty assets; no speedup is claimed.

`demo/qaoa_demo.json` records frozen angles, optimizer evaluation counts,
training time, feasible probability, conditional cost and best **feasible**
sample cost. Missing feasible samples produce null costs, not spurious wins.
A final shot budget is not the total quantum computation cost. Classical
objective evaluations are not cost-equivalent to simulator expectations.

Corrected local results (three COBYLA restarts, up to 200 evaluations per
restart, 4096 seeded final shots):

| Depth | Exact feasible probability | Best feasible cost | Gap to optimum | Training time |
| --- | --- | --- | --- | --- |
| 1 | 33.8% | -0.028117 | 2.22% | 205 s |
| 2 | 18.9% | -0.028432 | 1.13% | 478 s |
| 3 | 3.1% | -0.026374 | 8.28% | 605 s |

No depth sampled the exact optimum. Best-of-4096 random feasible sampling
achieved -0.028492, better than all three QAOA results in this seeded run.

On the retrospective 2023–2024 test, the classical optimal portfolio returned
121.31% cumulatively after modeled costs; the depth-two sampled portfolio
returned 126.01%. Their annualized volatility was 18.77% and 18.84%,
respectively. This isolated backtest difference does not establish quantum
advantage, especially given the exploratory design and single sampling seed.

`data/backtest.json` measures net return, volatility, drawdown and turnover
for frozen portfolios under monthly equal-weight rebalancing and 10 basis
points per absolute weight traded. This includes initial investment from
cash. Results are a single historical experiment, not investment advice.

## Qollab execution

Upload `main.py`, the `src/` tree and `demo/qaoa_demo.json`. Qollab need not
install this project or run uv. Runtime imports use the source tree; training
and data dependencies load only in local workflows. `main.py` submits one
frozen circuit as one job and never trains on the selected backend.

Local tests are not confirmation of Qollab or IonQ compatibility. Verify the
published project on Qollab separately before submission.

## Documentation and rights

[Criterion write-up](docs/criterion_summary.md) · [Disclosures](DISCLOSURES.md).
Code and documentation are MIT licensed. Market-data redistribution rights
are separate and have not been established by the code license.

This effort is supported by Qollab & IonQ.
