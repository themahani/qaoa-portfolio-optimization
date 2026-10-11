"""QAOA circuit for the portfolio Ising Hamiltonian and local angle training.

The cost Hamiltonian is the Ising model from :func:`qubo_to_ising`::

    H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j,

one qubit per asset. Each QAOA layer applies ``exp(-i*gamma*H_C)`` (RZ/RZZ
rotations) followed by the transverse-field mixer ``exp(-i*beta*sum(X))``
(RX rotations). Angles are pre-trained locally with COBYLA against exact
statevector expectations; the Qollab demo runs the frozen circuit as one job.
"""

from __future__ import annotations

import time

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

_STATEVECTOR_BACKEND = AerSimulator(method="statevector")


def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    gammas: list[float] | np.ndarray,
    betas: list[float] | np.ndarray,
    measure: bool = True,
) -> QuantumCircuit:
    """Build the QAOA circuit for one portfolio Ising instance.

    Parameters
    ----------
    h:
        Local fields, shape ``(N,)``.
    J:
        Symmetric coupling matrix with zero diagonal, shape ``(N, N)``.
    gammas, betas:
        One angle per QAOA layer (depth ``p = len(gammas)``).
    measure:
        Append a full measurement when True (demo/sampling mode);
        omit for statevector training.

    Returns
    -------
    QuantumCircuit
        ``H`` on all qubits, then per layer ``RZ(2*gamma*h)``,
        ``RZZ(2*gamma*J)`` on every coupled pair and ``RX(2*beta)``.
    """
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    gammas = list(gammas)
    betas = list(betas)
    if len(gammas) != len(betas):
        raise ValueError("gammas and betas must have equal length")
    n = h.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if J[i, j] != 0.0]
    circuit = QuantumCircuit(n, n if measure else 0)
    circuit.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            if h[i] != 0.0:
                circuit.rz(2.0 * gamma * h[i], i)
        for i, j in pairs:
            circuit.rzz(2.0 * gamma * J[i, j], i, j)
        circuit.rx(2.0 * beta, range(n))
    if measure:
        circuit.measure(range(n), range(n))
    return circuit


def all_ising_energies(
    h: np.ndarray, J: np.ndarray, offset: float = 0.0, block: int = 1 << 16
) -> np.ndarray:
    """Precompute the Ising energy of every spin configuration.

    Parameters
    ----------
    h:
        Local fields, shape ``(N,)`` with ``N <= 24``.
    J:
        Symmetric coupling matrix with zero diagonal.
    offset:
        Constant energy shift.
    block:
        Configurations evaluated per chunk.

    Returns
    -------
    np.ndarray
        Energies of all ``2^N`` configurations in integer order
        (bit ``i`` of the index = spin ``i``, ``0`` -> +1, ``1`` -> -1).
    """
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    n = h.shape[0]
    if n > 24:
        raise ValueError(f"N={n} exceeds the 24-variable limit")
    total = 1 << n
    energies = np.empty(total)
    for lo in range(0, total, block):
        hi = min(lo + block, total)
        bits = ((np.arange(lo, hi)[:, None] >> np.arange(n)) & 1).astype(np.float64)
        spins = 1.0 - 2.0 * bits
        energies[lo:hi] = (
            spins @ h + np.einsum("bi,ij,bj->b", spins, J, spins) / 2.0 + offset
        )
    return energies


def state_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    """Exact outcome probabilities of a measurement-free circuit.

    Parameters
    ----------
    circuit:
        QAOA circuit built with ``measure=False``.

    Returns
    -------
    np.ndarray
        ``|amplitude|^2`` per computational basis state (Qiskit ordering:
        outcome bit ``i`` of index ``s`` is qubit ``i``).
    """
    stripped = circuit.remove_final_measurements(inplace=False) or circuit
    snapshot = stripped.copy()
    snapshot.save_statevector()
    state = _STATEVECTOR_BACKEND.run(snapshot).result().get_statevector()
    return np.abs(np.asarray(state)) ** 2


def expected_energy(
    params: np.ndarray,
    h: np.ndarray,
    J: np.ndarray,
    energies: np.ndarray,
    depth: int,
) -> float:
    """Exact QAOA expected energy for one angle vector (optimizer target).

    Parameters
    ----------
    params:
        Concatenated ``[gammas, betas]`` of length ``2 * depth``.
    h, J:
        Ising fields and couplings.
    energies:
        Precomputed :func:`all_ising_energies` table.
    depth:
        Number of QAOA layers ``p``.

    Returns
    -------
    float
        ``<psi(params)| H_C |psi(params)>`` from exact probabilities.
    """
    gammas = params[:depth]
    betas = params[depth : 2 * depth]
    probs = state_probabilities(build_qaoa_circuit(h, J, gammas, betas, measure=False))
    return float(probs @ energies)


def train_qaoa(
    h: np.ndarray,
    J: np.ndarray,
    offset: float = 0.0,
    depth: int = 1,
    restarts: int = 3,
    maxiter: int = 300,
    seed: int = 0,
) -> dict:
    """Pre-train QAOA angles locally with multi-restart COBYLA.

    Parameters
    ----------
    h, J:
        Ising fields and couplings.
    offset:
        Constant energy shift (only shifts the reported energy).
    depth:
        Number of QAOA layers ``p``.
    restarts:
        Independent COBYLA restarts; the best is kept.
    maxiter:
        COBYLA iterations per restart.
    seed:
        RNG seed for the initial angles (gamma in [0, 2*pi), beta in [0, pi)).

    Returns
    -------
    dict
        ``gammas``, ``betas``, ``expected_energy``, ``evals`` and ``seconds``
        of the best restart.
    """
    h = np.asarray(h, dtype=float)
    J = np.asarray(J, dtype=float)
    energies = all_ising_energies(h, J, offset)
    rng = np.random.default_rng(seed)
    best: dict = {"expected_energy": float("inf")}
    start = time.perf_counter()
    evals = 0
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 2 * np.pi, depth),
                             rng.uniform(0, np.pi, depth)])
        result = minimize(
            fun=lambda p: expected_energy(p, h, J, energies, depth),
            x0=x0,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        evals += int(result.nfev)
        if result.fun < best["expected_energy"]:
            best = {"expected_energy": float(result.fun),
                    "gammas": [float(v) for v in result.x[:depth]],
                    "betas": [float(v) for v in result.x[depth:]]}
    best["evals"] = evals
    best["seconds"] = time.perf_counter() - start
    return best


def counts_to_portfolios(counts: dict[str, int], n: int) -> list[tuple[np.ndarray, int]]:
    """Decode a counts dict into ``(binary_portfolio, shots)`` pairs.

    Qiskit orders outcome bits little-endian (leftmost character is qubit
    ``N-1``) and separates classical registers with spaces; both are undone
    here so asset ``i`` maps to element ``i`` of the portfolio vector.

    Parameters
    ----------
    counts:
        Raw counts mapping outcome strings to shot numbers.
    n:
        Number of qubits/assets.

    Returns
    -------
    list[tuple[np.ndarray, int]]
        ``(portfolio, shots)`` sorted by shots, most frequent first.
    """
    decoded = []
    for key, shots in counts.items():
        bits = key.replace(" ", "")
        if len(bits) != n:
            raise ValueError(f"outcome {key!r} has {len(bits)} bits, expected {n}")
        portfolio = np.array([int(bits[n - 1 - i]) for i in range(n)], dtype=np.int8)
        decoded.append((portfolio, int(shots)))
    decoded.sort(key=lambda item: item[1], reverse=True)
    return decoded
