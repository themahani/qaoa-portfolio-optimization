# Video script

Target length 2 to 3 minutes, about 330 words of narration. Times are approximate.

| Time | On screen | Narration |
| --- | --- | --- |
| 0:00 | Title: "QAOA Portfolio Selection: how a quantum win disappeared under a fair test" | This is a portfolio selection project for the Quantum Impact theme. It is also a story about how we almost fooled ourselves. |
| 0:10 | List of the 20 assets; "choose 5, equal weights" | The task: choose five of twenty assets, equally weighted, to balance return against risk. Each asset is one qubit, so the circuit has twenty qubits. |
| 0:25 | The Qollab project page; press Run on the IonQ Forte 1 simulator | We trained a QAOA circuit locally and run it here on Qollab as one circuit, one job. This is IonQ's Forte 1 simulator. The run takes about a minute and a half at depth one. |
| 0:45 | Qollab output: the sampled portfolios and the comparison lines | The output lists the best portfolios it sampled, and compares them with the exact best portfolio and with classical methods. |
| 1:00 | The old results table, with "beats rounding" highlighted and then struck through | Our first results showed QAOA beating a standard classical shortcut at every depth. That was wrong, for two reasons. We were comparing portfolios with different amounts of capital invested, and we were counting samples that did not hold exactly five assets. |
| 1:25 | Right panel of the figure | With both fixed, three simple classical methods find the exact best portfolio, two of them in under a hundred evaluations. QAOA's best portfolios are one to eight percent worse. Even picking valid portfolios at random does better. |
| 1:50 | Left panel of the figure | Here is why. Only a third of the shots are valid portfolios at depth one, and three percent at depth three. On IonQ's noise-model simulator it is sixteen percent, then five, then two. Deeper circuits waste more shots, not fewer. |
| 2:15 | The five-point protocol | What we take from this is a checklist: the same objective and capital for every method, count invalid samples, include random sampling as a baseline, report training cost, and test on unseen data. |
| 2:35 | "Next step: a circuit that only produces valid portfolios" and the repo link | The next step is a circuit that can only produce valid portfolios. Nothing here ran on quantum hardware, and we claim no speedup. Code, tests and disclosures are in the repository. This effort is supported by Qollab and IonQ. |

## Recording notes

- Record the Qollab run live at depth 1 (92 seconds on the IonQ Forte 1 simulator) and cut the
  wait, or run it beforehand and show the finished output.
- The old results table is in the README at commit `6ae3f97`; the correction is commit `fa1e885`.
- The figure is `docs/figures/qaoa_vs_classical.png`. Show the two panels one at a time.
- The hackathon says the video is not scored on production quality. A screen recording with
  narration is enough.
