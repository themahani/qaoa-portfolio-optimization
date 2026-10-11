# QAOA Portfolio Optimization

Cardinality-constrained portfolio optimization with the Quantum Approximate
Optimization Algorithm (QAOA) in Qiskit — built for the Qollab × IonQ
Hackathon 2026 under the **Quantum Impact** theme.

The Markowitz mean-variance problem becomes NP-hard once you require exactly
$k$ out of $N$ assets (limits transaction costs, fund mandates). QAOA tackles
that QUBO with a hybrid quantum-classical loop; a brute-force exact solver on
$2^{20}$ states supplies ground truth for scoring.

## Setup

Python 3.14 is required to match the Qollab runtime. The environment is
managed with `uv`, and `pyproject.toml` is the single source of truth:

```bash
uv sync
uv run python scripts/pull_data.py
uv run python scripts/bench_classical.py
uv run python scripts/train_qaoa.py
uv run pytest
```

Qiskit is pinned to `>=2.5.2,<3.0.0` to match Qollab's runtime default, and
`qiskit-aer==0.17.2` matches Qollab's PQCA library pin. Data fetching uses
only `requests` against the keyless Yahoo Finance chart API (no `yfinance`
dependency). `requirements.txt` holds the Qollab-runtime-only pins.

## Data

`scripts/pull_data.py` pulls daily adjusted closes for a 20-asset universe
(2015–2024) and derives month-end log returns:

| Category | Tickers |
| --- | --- |
| US equity | SPY, QQQ, IWM, DIA |
| Sectors | XLK, XLF, XLV, XLE, XLI, XLP, XLU |
| Bonds | TLT, LQD, HYG |
| Commodities | GLD, USO |
| International | EFA, EEM |
| Real estate | VNQ |
| Crypto | BTC-USD |

20 tickers map one-to-one onto 20 qubits, leaving headroom under Qollab's
24-qubit built-in simulator ceiling. All three outputs are far below Qollab's
2 MB per-file cap (~10y daily ≈ hundreds of KB, monthly ≈ tens of KB).

```
data/prices_daily.csv        daily adjusted closes
data/returns_monthly.csv     month-end log returns (mu / Sigma source)
data/universe.json           ticker metadata + annualized mean/vol
data/classical_baseline.json exact optima + continuous reference (ground truth)
```

## Classical baselines

`scripts/bench_classical.py` builds the cardinality-constrained QUBO
(`N=20, k=5, q=0.5`, provably dominating penalty `A`) and scores it exactly —
`2^20 = 1,048,576` states enumerate in under a second:

| Baseline | True cost | Portfolio |
| --- | --- | --- |
| Exact (all states) | −0.0537 | SPY, QQQ, XLK, GLD, BTC-USD |
| Exact (feasible, \|x\|=5) | −0.0537 | same (penalty provably dominates) |
| Top-5 of continuous Markowitz | −0.0491 | QQQ, XLK, XLF, XLP, BTC-USD (+8.6% gap) |

The continuous long-only Markowitz (SLSQP) is the realistic quant reference;
its top-`k` rounding is what QAOA must beat, and the exact feasible optimum
is the ground truth it is scored against. Metrics: unpenalized Markowitz
cost, plus feasibility rate and best-sampled cost for QAOA runs.

## Layout

```
main.py                   Qollab entry point: one frozen circuit, one job
scripts/pull_data.py      Yahoo chart API pull -> prices/returns/universe
scripts/bench_classical.py QUBO baselines -> data/classical_baseline.json
scripts/train_qaoa.py     angle pre-training -> demo/qaoa_demo.json
src/qaoa_portfolio/       data loaders, QUBO/Ising, baselines, QAOA circuit
tests/                    panel, QUBO, circuit, demo checks (20 tests)
demo/qaoa_demo.json       frozen angles + metrics + references for main.py
```

## QAOA

`src/qaoa_portfolio/circuit.py` maps the QUBO to Ising (`x = (1−z)/2`) and
builds the depth-`p` circuit: `H` on all 20 qubits, then per layer
`RZ(2γh)` + `RZZ(2γJ)` on all 190 pairs (dense — the cardinality penalty
couples every pair) + `RX(2β)` mixer. `scripts/train_qaoa.py` pre-trains
angles locally with multi-restart COBYLA against exact statevector
expectations and seeded finite-shot sampling, freezing everything into
`demo/qaoa_demo.json` (angles, distribution metrics, references).

Trained depths (3 COBYLA restarts × 200 iters, exact statevector; 4096
seeded shots; optimum −0.0537, classical top-5 rounding −0.0491):

| Depth | P(feasible) | Best of 4096 shots | vs top-5 |
| --- | --- | --- | --- |
| p=1 | 1.5% | −0.0512 | beats rounding |
| p=2 | 65.7% | −0.0523 (2.5% gap to opt) | beats rounding |
| p=3 | 5.2% | −0.0503 | beats rounding |

p=2 is the demo default. p=3 trails p=2 under the fixed optimizer budget —
deeper is not automatically better, and we report it as measured. No depth
put mass on the exact optimum in 4096 shots (`P(opt) ≈ 0`); the claim is
beating the rounding heuristic at matched sampling cost, not optimality.

## Qollab demo

`main.py` is the Qollab entry point: it builds the frozen p-`depth` circuit,
submits **one circuit as one job** (the IonQ constraint — same pattern as
our QRC project), decodes counts into portfolios and scores them with the
unpenalized Markowitz cost beside the exact optimum and the classical
top-`k` rounding. No training happens on Qollab; Qiskit + NumPy only.

## License

MIT. See [LICENSE](LICENSE). Provenance, dependency licenses and references:
[DISCLOSURES.md](DISCLOSURES.md).
