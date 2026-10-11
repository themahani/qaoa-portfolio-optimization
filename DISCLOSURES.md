# Disclosures

For the Qollab × IonQ Hackathon 2026 (Quantum Impact theme). Everything below
is disclosed per the hackathon terms; judging covers what this team built
during the build window (Oct 9–11, 2026).

## AI assistance

Developed with AI coding assistance (OpenCode agent, Muse Spark 1.3),
with every result verified by the human authors: all 20 tests pass, the
benchmark numbers are reproduced by re-running the scripts, and the demo was
executed end-to-end on a local simulator. No AI-generated content is presented
as measured data; all tables report script outputs.

## Libraries and licenses

We are licensed to use every dependency below under its OSI-approved license.
No copyleft (GPL/AGPL) package is used anywhere in the dependency closure.

| Package | License | Used for | Ships to Qollab? |
| --- | --- | --- | --- |
| `qiskit >=2.5.2,<3.0.0` | Apache-2.0 | QAOA circuits | yes (`requirements.txt`) |
| `qiskit-aer ==0.17.2` | Apache-2.0 | local exact statevector training/simulation | no — Qollab provides Aer via its PQCA library entry |
| `numpy >=1.26` | BSD-3-Clause | numerics everywhere | runtime built-in, not pinned for Qollab |
| `scipy >=1.12` | BSD-3-Clause | SLSQP continuous baseline, COBYLA training | runtime built-in, not pinned for Qollab |
| `pandas >=2.0` | BSD-3-Clause | data handling | yes (`requirements.txt`) |
| `requests >=2.31` | Apache-2.0 | data pull only | no — demo ships CSVs |
| `matplotlib >=3.8` | Matplotlib (BSD-compatible PSF-based) | figures only | no — demo ships pre-rendered figures |
| `pytest >=9` (+ `pluggy`, `pygments`, `iniconfig`, all MIT) | MIT | tests only | no |
| `hatchling` | MIT | build backend only | no |

Python 3.14 itself is under the PSF License. Our code and docs are MIT
licensed (see `LICENSE`).

## Data provenance

- Price data: daily adjusted closes for 20 tickers (SPY, QQQ, IWM, DIA, XLK,
  XLF, XLV, XLE, XLI, XLP, XLU, TLT, LQD, HYG, GLD, USO, EFA, EEM, VNQ,
  BTC-USD), Jan 2015–Dec 2024, fetched from Yahoo Finance's keyless public
  chart API (`scripts/pull_data.py`, `requests` only — no `yfinance`).
- Derived artifacts committed for reproducibility: `data/prices_daily.csv`,
  `data/returns_monthly.csv` (month-end log returns — the mu/Sigma source),
  `data/universe.json`. Yahoo data is used for non-commercial research under
  Yahoo's terms; only our derived datasets are redistributed, under MIT
  alongside the code.
- The fetch pattern (chart-API + `requests`, no scraper) was adapted from our
  own MIT-licensed QRC hackathon repo (same team, same event).

## References followed (no code copied)

The implementation was written from the mathematical formulations in these
works; no external code, circuit, or repository was copied or forked:

- H. Markowitz, "Portfolio Selection," *J. Finance* 7(1), 1952 —
  mean-variance objective and the cardinality-constrained QUBO.
- E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization
  Algorithm," arXiv:1411.4028 (2014) — the QAOA ansatz and angle structure.
- J. Hodson et al., "Portfolio rebalancing experiments using the Quantum
  Alternating Operator Ansatz," arXiv:1911.05296 (2019) — QAOA applied to
  portfolio problems; problem-context precedent, implementation ours.

## Prior / external code

None, except the team's own MIT-licensed fetch pattern noted above. The QUBO
construction, Ising mapping, brute-force and continuous baselines, QAOA
circuit, training loop, demo entry point, tests and docs were all written
during the hackathon build window for this project.

## Attribution

This effort is supported by Qollab & IonQ.
