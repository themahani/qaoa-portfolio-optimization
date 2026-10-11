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
data/prices_daily.csv     daily adjusted closes
data/returns_monthly.csv  month-end log returns (mu / Sigma source)
data/universe.json        ticker metadata + annualized mean/vol
```

## Layout

```
scripts/pull_data.py      Yahoo chart API pull -> prices/returns/universe
src/qaoa_portfolio/       data loaders + mean/cov helpers (QUBO next)
tests/test_data.py        panel shape, coverage, 2 MB cap checks
demo/                     Qollab demo payload (TBD)
```

## License

MIT. See [LICENSE](LICENSE).
