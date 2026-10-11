# Written summary — one section per judging criterion

## Novelty and creativity

Portfolio optimization with QAOA has been tried before, so our novelty is not
the idea itself. It is the honesty and completeness of the comparison around
it.

Most QAOA demonstrations compare against weak or vaguely described classical
methods, which makes it impossible to tell whether the quantum part did
anything useful. We did the opposite. We built the exact answer first: every
one of the 1,048,576 possible portfolios was evaluated, so the true optimum is
known, not estimated. We also built the method a working quant would actually
use — a continuous Markowitz optimization rounded down to the five best
assets. QAOA is scored against both of these, on the same numbers, with the
same cost function. That kind of ground-truth benchmarking for a 20-qubit
finance QUBO, published openly with all code and data, is something we have
not seen elsewhere.

We also report what did not work. A three-layer QAOA circuit scores worse
than the two-layer one under our fixed training budget, and no circuit depth
ever sampled the exact optimum in 4,096 shots. Many write-ups would quietly
drop those facts. We kept them, because a benchmark that only reports wins is
advertising, not science. The penalty construction is a smaller novelty: we
prove the cardinality penalty dominates any possible objective gain, so the
exact optimum is guaranteed feasible — and then both the penalized and the
feasible-only searches land on the same portfolio, confirming the proof in
practice.

## Technical execution

Everything runs, and everything is checked. The project is built with Qiskit
and runs on Qollab's free simulators: the demo submits exactly one circuit as
one job, which is the hard constraint on IonQ's remote simulators, and the
same code path was executed end-to-end on a local simulator in our test
suite. Twenty automated tests cover the data panel (shape, date coverage, the
2 MB file cap), the QUBO construction (energy matches the textbook objective
term by term), the QUBO-to-Ising mapping (energies preserved exactly under
the substitution), the brute-force solver (agrees with naive enumeration),
and the demo itself (payload fields, circuit statistics, a full run of the
entry point).

Two bugs were caught by these tests before they could reach the results — a
missing risk term on the QUBO diagonal and an off-by-one in the Ising field
formula — which is the point of writing the tests first. The pipeline is
seeded and reproducible: the same commands produce the same angles, the same
energies, and the same tables. Energy tables are precomputed once and reused,
so training is minutes rather than hours, and the demo carries only frozen
numbers — no training, no heavy libraries, just Qiskit and NumPy on Qollab's
side.

## Impact potential

The direct impact is a reusable, honest template for asking "did the quantum
subroutine help?" in finance. Anyone can take this repo, swap in their own
assets or their own variational circuit, and get an answer measured against
the exact optimum instead of a strawman. That matters because finance is full
of quantum claims that cannot be checked; a public harness where the
classical side is exact raises the bar for all of them.

The result itself is modest and we state it plainly: at 20 assets, exact
classical enumeration takes half a second and beats everything, so nobody
should run this on a quantum computer to save time today. What QAOA does do
is beat the realistic classical shortcut — the top-5 rounding of the
continuous solution — at a matched sampling budget, landing within 2.5% of
the true optimum at depth two. That is a narrow claim, but it is a real one:
a shallow quantum heuristic outperforming the standard classical heuristic on
real market data with the same number of samples. For students and
practitioners, the project also works as a complete worked example of the
full hybrid loop — data, QUBO, exact baseline, variational training,
single-job deployment — with every step runnable and every number
reproducible.

## Documentation quality

The repository is written to be read by strangers. Every function carries a
plain-language docstring explaining what it does, what it takes, and what it
returns. The README walks through setup (four commands), the data (what each
ticker is and why it is there), the baselines (with the actual numbers), the
QAOA method, and the demo — including the results table with the depths that
underperformed. DISCLOSURES.md lists every dependency with its license,
states the data provenance, and names the papers the design follows. Figures
and tables in the project page label which numbers came from local simulation
and which from the Qollab run, so a judge never has to guess what ran where.
A newcomer can clone the repo, run the test suite, and reproduce every claim
in this summary in under half an hour.
