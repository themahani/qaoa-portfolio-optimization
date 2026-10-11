# QAOA resource analysis

## What was compiled

`scripts/analyze_resources.py` rebuilds the frozen twenty-asset circuits and
transpiles them using Qiskit to `h`, `rz`, `cx`, with optimization level zero,
seed zero and no coupling map. It also checks synthetic dense circuits with
4, 8, 12, 16, 20 and 24 assets at p=2. Assertions verify the formulas below.
Outputs are saved in `data/circuit_resources.json`.

This basis is universal when arbitrary RZ angles are allowed. It is not a
finite fault-tolerant gate set; rotation synthesis precision and T counts
are outside this analysis. Nor is it IonQ's native gate set. Routing, native
compilation and error correction can change physical costs.

## Derivation

The circuit uses N initial Hadamards. Each of p layers contains N field RZ
rotations, M=N(N−1)/2 pairwise RZZ rotations, and N mixer RX rotations.
The cardinality penalty makes every pair coupled even if some financial
covariances are zero.

- `RZZ(theta)` decomposes into `CX – RZ(theta) – CX`.
- `RX(theta)` decomposes into `H – RZ(theta) – H`.
- Field `RZ` gates and initial `H` gates are already in the basis.

With dense nonzero coefficients and no gate optimization:

| Resource | Exact count |
| --- | --- |
| Qubits | N |
| H gates | N(1+2p) |
| RZ gates | p(M+2N) |
| CX gates | 2pM = pN(N−1) |
| Total unitary gates | N+p(3M+4N) |
| Measurements per shot | N |

Measurements are excluded from the unitary total. Special angles,
zero coefficients or optimization can reduce counts.

## Measured twenty-asset circuits

| Layers | H | RZ | CX | Unitary total | Transpiled depth including measurements |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 60 | 230 | 380 | 670 | 117 |
| 2 | 100 | 460 | 760 | 1,320 | 181 |
| 3 | 140 | 690 | 1,140 | 1,970 | 245 |

These depths differ from the original RZZ/RX circuit depths 41, 63 and 85
reported on Qollab. Those counts describe different bases.

## Measured scaling at p=2

| Assets | Qubits | H | RZ | CX | Unitary total | Transpiled depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 20 | 28 | 24 | 72 | 37 |
| 8 | 8 | 40 | 88 | 112 | 240 | 73 |
| 12 | 12 | 60 | 180 | 264 | 504 | 109 |
| 16 | 16 | 80 | 304 | 480 | 864 | 145 |
| 20 | 20 | 100 | 460 | 760 | 1,320 | 181 |
| 24 | 24 | 120 | 648 | 1,104 | 1,872 | 217 |

The observed p=2 logical depths are linear in N for this emitted schedule
without routing. Gate work remains quadratic; logical depth alone is not
the runtime of either a physical backend or a statevector simulator.

## Complexity of the full workflow

Let S be shots per expectation evaluation and K the number of optimizer
evaluations. These are distinct from the final demo's shot budget.

| Stage | Work / memory scaling | Assumptions |
| --- | --- | --- |
| Circuit construction | O(pN²) work, O(pN²) stored operations | Dense objective |
| One circuit shot | O(pN²) gate work | Not elapsed time |
| Sampling S shots | O(SpN²) gate work | Repeated execution; ideal probability-based simulation may differ |
| Classical portfolio scoring | O(SN²) work | Dense covariance; upper bound before deduplication |
| Shot-based training | O(KSpN²) quantum gate work | Does not bound convergence or required p, K, S |
| Actual local exact training | O(KpN²2^N) work upper bound | Gates act on exponentially many amplitudes |
| Statevector/probability table | O(2^N) numerical memory | Circuit storage is additional |
| Energy-table precomputation | O(N²2^N) work | Chunked enumeration; cache reused |

Our implementation trains against exact simulator expectations, not S-shot
hardware evaluations. The quantum rows are resource models, not measured
physical runtime bounds. Data preparation, classical optimizer overhead,
sorting, compilation, queueing and communication add costs.

Classical full enumeration checks 2^N states. Feasible-only enumeration
instead checks C(N,k) selections; for fixed k=5 this is O(N^5) candidates,
not an exponential count in N. Selecting a growing fraction of assets gives
combinatorial growth. Strong classical methods need not enumerate either
space, as the measured greedy and relaxation results show.

Linear qubit scaling is a compact representation property, not a proof of
asymptotic optimization advantage. QAOA does not guarantee that fixed p and
polynomial K and S find the optimum. Current results show no speedup or
quality improvement over the classical competitors.

## Reproduce

From the repository root:

```bash
uv sync
uv run python scripts/analyze_resources.py
```
