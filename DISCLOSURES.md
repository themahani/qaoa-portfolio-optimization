# Disclosures

## AI assistance

Code and documentation were developed with OpenCode AI assistance. The agent
in this session identifies itself as OpenAI GPT-6.1 Sol Fast. The previous
disclosure naming Muse Spark 1.3 was incorrect. Automated checks and local
script executions verify specific behaviors; they do not establish human
review of every line or platform compatibility.

## Code and dependency licenses

Our source code and documentation are MIT licensed. Direct dependencies are
Qiskit and Qiskit Aer (Apache-2.0), NumPy, SciPy and pandas (BSD-3-Clause),
requests (Apache-2.0), matplotlib (its PSF-based license), pytest (MIT), and
hatchling (MIT). Dependency licenses and notices remain applicable; see each
distribution for the authoritative terms. `uv.lock` records resolved versions.
We do not claim to have audited every transitive dependency. The project does
not import typing-extensions directly; dependencies may require it.

Local fetching, optimization and tests use libraries that the Qollab demo
does not directly import. The demo uses Qiskit and NumPy from source-tree
imports, without an editable install or uv on Qollab.

## Data provenance and rights

Yahoo Finance's public chart endpoint supplied daily adjusted close prices
for the twenty tickers listed in the README over 2015–2024. Monthly simple
returns, metadata and optimization results are derived from those prices.
The fetch pattern was adapted from the team's MIT-licensed QRC hackathon
repository.

Public accessibility does not grant an MIT license to market data. The MIT
license applies to our code and documentation, not Yahoo data or third-party
data-provider rights. Permission to redistribute the bundled market-data
snapshot has not been established here. Users and submitters must check the
applicable terms before redistributing or using it; derived data should not
be assumed exempt from those terms.

## Methods and prior work

The methods are standard: Markowitz mean–variance selection, binary-to-Ising
mapping, QAOA, continuous relaxation, enumeration and swap local search.
Background references include Markowitz, “Portfolio Selection” (1952), and
Farhi, Goldstone and Gutmann, “A Quantum Approximate Optimization Algorithm,”
arXiv:1411.4028 (2014). These are background references, not claims that the
implementation reproduces a particular published benchmark. No claim of
first-of-its-kind implementation is made.

This project was created separately from the team's QRC project during this
hackathon session. Local simulation results are not quantum hardware results.
Successful Qollab execution must be verified separately. This is a research
prototype, not investment advice or evidence of quantum advantage.

This effort is supported by Qollab & IonQ.
