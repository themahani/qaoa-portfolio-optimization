# Written summary — judging criteria

## Novelty and creativity

This project does not introduce portfolio QAOA or claim to be the first
implementation. It explores how to evaluate quantum portfolio selection
without confusing invalid portfolios, different investment budgets, or a
weak classical comparison with evidence of quantum advantage.

Each valid solution chooses five assets and assigns one fifth of the capital
to each. The exact solver, continuous relaxation, classical heuristics and
QAOA are scored against that same investment objective. We count invalid
quantum samples, but do not let their costs enter the comparison. We publish
negative results rather than selecting only settings where quantum appears
to win. After correcting the investment model, continuous rounding already
finds the exact optimum for our training instance. That is an important
result, not something to conceal.

## Technical execution

The implementation uses Qiskit to construct the circuit and maps the binary
portfolio objective to an Ising Hamiltonian. An exhaustive classical search
checks every binary state. A separate search over valid five-asset portfolios
checks that the penalty enforces the constraint. Tests compare the QUBO,
Ising model and direct equal-weight investment objective.

Training happens locally using Aer and COBYLA. The demo submits one frozen
circuit as one job and interprets measured bitstrings as selected assets.
Its imports do not eagerly load the data or training dependencies. Automated
checks exercise local execution and source-tree imports without installing
the project. The demo was then run on Qollab's IonQ Forte 1 noise-model
simulator at all three depths with 4,096 shots each; it completed without
errors in 92 to 224 seconds per run. Under that noise model the share of
valid portfolios fell from 34% to 16% at depth one and from 20% to 5% at
depth two. The runs are recorded in `data/ionq_runs.json`.

The evaluation records feasible-sample rates and best feasible costs. It also
includes uniform feasible sampling, greedy selection, swap search from the
rounded solution, and multi-start random swap search. Classical objective
counts and timings are recorded. Quantum optimizer evaluations and training
time are separate from the final shot budget: a simulator expectation
evaluation is not equivalent in cost to a classical objective evaluation.

## Impact potential

The useful outcome is a reproducible evaluation workflow, not a trading
recommendation. At this size, exact classical optimization is inexpensive.
Our corrected rounded baseline reaches the optimum, so this instance gives
no reason to replace that method with a quantum computer.

Return estimates and covariance use only 2015–2020 observations. The later
2021–2022 interval is reserved rather than used to tune the current fixed
configuration. A separate backtest evaluates frozen portfolios on 2023–2024
data, with monthly equal-weight rebalancing and an explicit trading-cost
assumption. This separates the question of solving an estimated objective
from the question of investment performance on unseen observations. One
historical split does not establish generalization or financial value.

## Documentation quality

The README explains the investment convention, data preparation, chronological
split, exact answer and classical alternatives. Functions document their
inputs and outputs. The disclosures distinguish our MIT code license from
third-party market-data rights and identify AI assistance.

The scripts regenerate the data-derived artifacts, classical comparisons,
trained circuit payload and held-out backtest. Saved results identify the
training window and retain failed or infeasible outcomes. Every table and
the figure state whether a number comes from the noiseless local simulation
or from IonQ's simulator on Qollab. Neither is a quantum hardware result.
