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

## Results

All optimization rows use the same twenty-asset universe, five selected
assets, equal investment weights, q=0.5, and 2015–2020 training estimates.
**Lower objective cost is better; objective costs are not investment returns.**
Only feasible portfolios enter the cost comparison. The relative gap is
`100 * (cost - exact_cost) / abs(exact_cost)`.

### Optimization: classical, local QAOA and IonQ simulator

| Method | Execution environment | Candidate budget | Best feasible cost | Gap to optimum |
| --- | --- | --- | ---: | ---: |
| Exact enumeration | Local NumPy | 1,048,576 binary states | -0.028756 | 0.00% |
| Exact feasible enumeration | Local NumPy | 15,504 five-asset portfolios | -0.028756 | 0.00% |
| Continuous relaxation + rounding | Local SciPy | One solve + rounding | -0.028756 | 0.00% |
| Greedy selection | Local NumPy | 91 objective evaluations | -0.028756 | 0.00% |
| Random starts + swaps | Local NumPy | 16 starts; 6,032 objective evaluations | -0.028756 | 0.00% |
| Rounding + swaps | Local NumPy | One start; 77 objective evaluations | -0.028756 | 0.00% |
| Random feasible sampling | Local NumPy | 4,096 feasible candidates | -0.028492 | 0.92% |
| QAOA p=1 | Local ideal Aer + seeded sampling | 4,096 shots after training | -0.028117 | 2.22% |
| QAOA p=2 | Local ideal Aer + seeded sampling | 4,096 shots after training | -0.028432 | 1.13% |
| QAOA p=3 | Local ideal Aer + seeded sampling | 4,096 shots after training | -0.026374 | 8.28% |
| QAOA p=2 | Qollab → IonQ Forte 1 noise-model simulator | 4,096 shots; frozen local angles | -0.028117 | ≈2.22% |

The IonQ cost was printed to six decimal places; its gap is therefore
approximate. It is a **remote simulation result, not a quantum hardware run**.
The selected backend's advertised capacity was 29 qubits; our circuit used
20 qubits, not 29.

The optimum is **SPY, QQQ, DIA, XLK, BTC-USD**, with 20% invested in each.
Several inexpensive classical methods reach it. Random feasible sampling
beats all three local QAOA depths and the reported remote run in these
single-seed/single-run comparisons. No speedup or quantum advantage is claimed.
The earlier claim of beating rounding used inconsistent capital scaling and
included infeasible samples; it is withdrawn.

### Feasibility and execution cost

Local training used three COBYLA restarts with up to 200 evaluations per
restart. Exact feasible probabilities come from ideal statevectors; observed
rates come from 4,096 sampled outcomes. These are different quantities.

| Run | Exact ideal feasible probability | Observed feasible rate | Circuit depth | RZZ gates before backend compilation | Local training evaluations | Local training time | Reported remote circuit execution time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local p=1 | 33.8% | 34.3% | 41 | 190 | 441 | 205.4 s | — |
| Local p=2 | 18.9% | 20.3% | 63 | 380 | 600 | 478.1 s | — |
| Local p=3 | 3.1% | 3.2% | 85 | 570 | 600 | 604.7 s | — |
| IonQ Forte 1 simulator p=2 | 18.9% reference; not a remote probability estimate | 5.4% | 63 | 380 | Reused local p=2 angles | No retraining on Qollab | 155.0 s |

The remote p=2 run retained substantially fewer valid portfolios than the
local ideal run: 5.4% versus 20.3% observed. IonQ's remote simulator applies
a device noise model, whereas the local reference is ideal. This single run
does not isolate the effects of noise, compilation and sampling. Raw counts
and backend characterization were not supplied with the pasted output.

The 155.0 s value is Qollab's reported circuit execution time (155,000 ms),
not total elapsed time including package loading, initialization or queueing.
It also excludes the 478.1 s of local angle training. Classical enumeration
took approximately 0.59 s; greedy objective scoring took approximately
0.40 ms in the saved local benchmark. These are different computations and
indicative timings, not a controlled cross-platform runtime comparison.
A classical objective evaluation is not cost-equivalent to a simulator
expectation evaluation or a shot.

### Best feasible portfolios

| Result | Selected assets, each at 20% | Objective cost |
| --- | --- | ---: |
| Exact / rounding / greedy / swap-search optimum | SPY, QQQ, DIA, XLK, BTC-USD | -0.028756 |
| Random feasible sampling | SPY, QQQ, XLK, XLV, BTC-USD | -0.028492 |
| Local QAOA p=1 | QQQ, DIA, XLK, XLU, BTC-USD | -0.028117 |
| Local QAOA p=2 | SPY, QQQ, XLK, XLF, BTC-USD | -0.028432 |
| Local QAOA p=3 | SPY, XLK, XLV, EEM, BTC-USD | -0.026374 |
| IonQ Forte 1 simulator QAOA p=2 | QQQ, DIA, XLK, XLU, BTC-USD | -0.028117 |

The remote p=2 winner happens to equal the local p=1 winner; those are separate
runs of different trained circuits, not duplicated measurements.

### Top five feasible portfolios from the reported IonQ run

| Rank | Selected assets | Objective cost | Observed shots |
| --- | --- | ---: | ---: |
| 1 | QQQ, DIA, XLK, XLU, BTC-USD | -0.028117 | 1 |
| 2 | QQQ, XLK, XLU, EEM, BTC-USD | -0.027350 | 1 |
| 3 | QQQ, XLK, XLV, VNQ, BTC-USD | -0.027151 | 1 |
| 4 | QQQ, XLK, XLU, TLT, BTC-USD | -0.027088 | 1 |
| 5 | QQQ, XLK, LQD, EEM, BTC-USD | -0.026864 | 1 |

These are ranked by objective cost, not sampling frequency. Each appeared
once, so the table does not establish stable portfolio-selection probabilities.

### Retrospective held-out backtest: 2023–2024

Frozen portfolios are rebalanced monthly to equal weights, with 10 basis
points charged per absolute weight traded, including initial investment from
cash. Return is cumulative over two years; volatility is annualized and
drawdown is measured at monthly observation points.

| Portfolio source | Net cumulative return | Annualized volatility | Maximum drawdown | Total absolute weight turnover |
| --- | ---: | ---: | ---: | ---: |
| Classical exact / rounding / greedy / swap-search optimum | 121.31% | 18.77% | -6.81% | 1.914 |
| Random feasible sampling | 111.53% | 18.06% | -6.84% | 1.942 |
| Local QAOA p=1 | 108.68% | 18.28% | -7.67% | 2.004 |
| Local QAOA p=2 | 126.01% | 18.84% | -6.69% | 1.940 |
| Local QAOA p=3 | 91.26% | 17.57% | -6.90% | 1.991 |

No separate remote backtest was run. The reported remote winner is identical
to the local p=1 portfolio, so the existing p=1 backtest describes that same
allocation—not an independently measured IonQ investment result.
The local p=2 portfolio's higher realized return does not establish quantum
advantage: this is one exploratory historical split and one sampling seed.
These results are not investment advice.

### Result provenance

- `data/classical_baseline.json`: classical scores, selected assets, budgets
  and local timings.
- `demo/qaoa_demo.json`: ideal local probabilities, frozen angles, training
  costs and seeded local sampling results. It does **not** contain IonQ counts.
- `data/backtest.json`: retrospective portfolio-performance calculations.
- IonQ Forte 1 simulator rows: participant-provided Qollab output from the
  successful p=2, 4,096-shot run. The display identified the backend as
  `ionq_simulator`; the participant identified the selected option as Forte 1.

## Qollab execution

Upload `main.py`, the `src/` tree and `demo/qaoa_demo.json`. Qollab need not
install this project or run uv. Runtime imports use the source tree; training
and data dependencies load only in local workflows. `main.py` submits one
frozen circuit as one job and never trains on the selected backend.

A participant successfully ran the p=2 demo on Qollab's IonQ Forte 1 simulator
with 4,096 shots; its results are included above. This verifies that reported
configuration, not every backend or circuit depth. After publishing the final
version, verify it again from another account or a logged-out browser.

## Documentation and rights

[Criterion write-up](docs/criterion_summary.md) · [Disclosures](DISCLOSURES.md).
Code and documentation are MIT licensed. Market-data redistribution rights
are separate and have not been established by the code license.

This effort is supported by Qollab & IonQ.
