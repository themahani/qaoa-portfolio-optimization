# Project page text

Paste-ready text for the hackathon project page. Theme tag: **Quantum Impact**.

## Title

QAOA Portfolio Selection: how a quantum "win" disappeared under a fair test

## One-line summary

We built a 20-qubit QAOA portfolio selector in Qiskit, found it beating a classical baseline,
traced that win to two evaluation mistakes, and turned the correction into a test that any
quantum optimization claim should pass.

## What it does

The task is to choose 5 of 20 assets (19 ETFs and Bitcoin), equally weighted, to balance
expected return against risk. Each asset is one qubit. A QAOA circuit, trained locally, is run
once on the backend you pick on Qollab; the measured bitstrings are read as portfolios and
scored against the exact best portfolio and several classical methods.

## The story

**1. An apparent win.** Our first results showed QAOA beating the standard classical shortcut
(solve the problem with fractional weights, then round to five assets) at every circuit depth.

**2. Why it was wrong.** Two mistakes produced that result. Portfolios were being compared at
different amounts of invested capital, and samples that did not hold exactly five assets were
allowed into the comparison. We corrected both, added tests for them, and withdrew the claim.

**3. What a fair test shows.** With every method scored on the same investment objective and
only valid portfolios counted:

- Rounding, greedy selection and a swap search all find the exact best portfolio, two of them
  in under 100 evaluations.
- QAOA's best valid portfolios are 1.1% to 8.3% worse, after 441 to 600 training evaluations
  and 4,096 shots per depth.
- Drawing 4,096 valid portfolios at random beats every QAOA depth.

**4. Why QAOA struggles here.** The five-asset rule is the bottleneck. On a noiseless
simulation only 34% of shots are valid portfolios at depth 1, 20% at depth 2 and 3% at depth 3.
On IonQ's Forte 1 noise-model simulator those fall to 16%, 5% and 2%, close to the 1.5% that
random bitstrings would give. Deeper circuits do worse, not better.

**5. What to take away.** A short protocol for judging quantum optimization results in
finance, and one clear direction for improvement.

![QAOA against classical baselines](figures/qaoa_vs_classical.png)

## The protocol

1. Score every method on the same objective with the same invested capital.
2. Count invalid samples, and keep them out of the cost comparison.
3. Include random valid sampling as a baseline, with the same number of candidates.
4. Report training cost separately from the final shot count.
5. Test the chosen portfolios on data the optimizer never saw.

## How to run it

Open the project on Qollab, pick a backend, and run. Each run submits one circuit as one job.

| Field | Meaning |
| --- | --- |
| `shots` | Number of shots (default 4,096) |
| `depth` | QAOA depth: 1, 2 or 3 |

On the IonQ Forte 1 simulator the three depths took 92, 155 and 224 seconds.

## What ran where

- **On Qollab:** the frozen QAOA circuit, on the IonQ Forte 1 noise-model simulator, at all
  three depths (`data/ionq_runs.json`).
- **Locally:** angle training (Qiskit Aer), the exact search over all 1,048,576 bitstrings, the
  classical baselines and the 2023–2024 backtest.
- **Not done:** nothing ran on quantum hardware.

## Limits

- One problem instance, one training recipe and one sampling seed or run per setting.
- At 20 assets an exact classical search takes under a second, so no speedup is possible or
  claimed.
- The backtest is a single historical split and is not evidence of investment performance.
- The clearest next step is a circuit that can only produce valid portfolios, so that no shots
  are wasted on the constraint. We have not built it.

## Links

- Repository: https://github.com/themahani/qaoa-portfolio-optimization
- Written summary by judging criterion: `docs/criterion_summary.md`
- Disclosures: `DISCLOSURES.md`

This effort is supported by Qollab & IonQ.
