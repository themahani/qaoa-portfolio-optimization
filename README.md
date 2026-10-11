# QAOA Portfolio Optimization

**Twenty assets. Twenty qubits. A fair test of quantum portfolio selection.**

## What is it?

A hybrid quantum–classical portfolio selector built with Qiskit for the
**Quantum Impact** theme. It selects five equally weighted holdings from
nineteen ETFs and Bitcoin, using real market data to balance expected return
against risk. Every proposed portfolio is checked against the exact answer
and strong classical alternatives.

## Why it matters

Choosing a small portfolio means weighing return, risk and diversification
across many possible asset combinations. Quantum optimization offers a way
to generate candidates without explicitly enumerating every portfolio.
Our project makes that idea testable: it combines a compact quantum encoding,
classical angle training and a reproducible evaluation with real data.

The contribution is an honest, runnable benchmark—not a promised quantum
win. Classical heuristics reach the optimum on this instance. We show where
QAOA loses quality, spends training effort and wastes measurements on invalid
portfolios, giving future implementations concrete targets for improvement.

## How the quantum part works

Each asset is one qubit. A measured **1** selects it; **0** leaves it out.
The portfolio objective and a five-holding penalty become an Ising
Hamiltonian. QAOA alternates a cost layer, which changes phases according to
that objective, with a mixer that changes candidate amplitudes. A classical
optimizer learns the angles locally; Qollab runs the frozen circuit once
and scores its measured portfolios.

**The architectural advantage is compact encoding:** N assets require exactly
N qubits, with no ancillary register in this implementation. The state space
contains 2^N possible selections, but a measurement returns only one selection;
it does not reveal or score all portfolios at once. This is a resource property,
not evidence of computational speedup.

### Resource scaling, checked by transpilation

We compiled the trained circuits into the universal gate set **Hadamard (H),
RZ rotations and controlled-NOT (CX)**, with no routing and optimization level
zero. These are logical gate counts, not IonQ native-gate counts.

| QAOA layers | Qubits | H gates | RZ gates | CX gates | Total unitary gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 60 | 230 | 380 | 670 |
| 2 | 20 | 100 | 460 | 760 | 1,320 |
| 3 | 20 | 140 | 690 | 1,140 | 1,970 |

For N assets and p layers, **qubits scale as O(N)** and the dense circuit's
**gate work scales as O(pN²)**. The penalty couples every asset pair, so linear
qubit scaling does not mean linear runtime. With S shots, quantum gate work
is O(SpN²); actual latency depends on parallelism, backend compilation and
execution. Local exact statevector training instead has an exponential
O(pN²2^N) per-evaluation upper bound. QAOA has no established speedup here.

See the [resource analysis](docs/resource_analysis.md) for the decomposition,
training costs and reproducible scaling check.

## What we measured

All methods use the same 2015–2020 estimates and equal investment weights.
Each quantum result below uses 4,096 shots. Lower objective cost is better;
only valid five-asset portfolios count.

| Method | Best feasible cost | Gap to optimum | Observed feasible rate | Remote execution time |
| --- | ---: | ---: | ---: | ---: |
| Classical exact / rounding / greedy / swap search | -0.028756 | 0% | Feasible output | — |
| Classical random feasible sampling | -0.028492 | 0.92% | 100% | — |
| Local ideal QAOA, p=2 | -0.028432 | 1.13% | 20.3% | — |
| IonQ Forte 1 simulator, p=1 | -0.027485 | ≈4.42% | 16.3% | 92 s |
| IonQ Forte 1 simulator, p=2 | -0.028117 | ≈2.22% | 5.4% | 155 s |
| IonQ Forte 1 simulator, p=3 | -0.026171 | ≈8.99% | 2.0% | 224 s |

Remote times exclude loading, queueing and local training. These are
**noise-model simulator runs, not hardware results**. They used 20 qubits on
a simulator with a 29-qubit capacity. One run per setting does not establish
a statistical advantage. [Full results and backtest](docs/results.md).

## Run it on Qollab

Open the project, select a **free simulator**, and run with `shots=4096` and
`depth=1`, `2` or `3` (default: `2`). The demo submits one circuit as one job,
then prints selected assets, costs and the classical comparison. No uv,
installation of this repository as a package, or training loop is required
on Qollab.

The three depths have been run successfully on Qollab's free **IonQ Forte 1
simulator**. For local reproduction:

```bash
uv sync
uv run python scripts/bench_classical.py
uv run python scripts/train_qaoa.py
uv run python scripts/analyze_resources.py
uv run pytest
```

**Dependencies:** Python 3.14, Qiskit and NumPy for the demo; Aer and SciPy
for local training, pandas and requests for data, matplotlib for figures.
Local versions are recorded in `pyproject.toml` and `uv.lock`.

## Project and demo

**Public Qollab project:** link to be added; confirm public MIT publication
before submission. **Demo video:** add a public 2–3 minute recording that
opens without signing in. Successful simulator runs are verified; publication
and video availability are not yet confirmed here.

Code and documentation are [MIT licensed](LICENSE).
[Disclosures](DISCLOSURES.md) explain data provenance and separate market-data
rights. This is a research prototype, not investment advice.

This effort is supported by Qollab & IonQ.
