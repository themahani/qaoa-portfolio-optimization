"""Feasible classical sampling and swap-search portfolio baselines."""

import time

import numpy as np


def compare_heuristics(mu: np.ndarray, covariance: np.ndarray, q: float,
                       k: int, rounded: np.ndarray, samples: int = 4096,
                       starts: int = 16, seed: int = 0) -> dict:
    """Benchmark random selection, greedy selection and multi-start swaps.

    Parameters
    ----------
    mu, covariance : np.ndarray
        Training-only simple-return moments.
    q : float
        Risk aversion.
    k : int
        Number of equally weighted selected assets.
    rounded : np.ndarray
        Feasible continuous-relaxation rounding, used as a search start.
    samples, starts : int
        Random candidate budget and number of local-search starts.
    seed : int
        Random generator seed.

    Returns
    -------
    dict
        Best cost, portfolio indices, objective evaluations and elapsed
        seconds per method. Local-search budgets are reported, not hidden.
    """
    rng = np.random.default_rng(seed)
    n = len(mu)
    results = {}
    evaluations = 0

    def cost(x: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        w = x / k
        return float(q * w @ covariance @ w - mu @ w)

    def swaps(x: np.ndarray) -> np.ndarray:
        x = x.copy()
        value = cost(x)
        while True:
            best, best_value = x, value
            for i in np.flatnonzero(x):
                for j in np.flatnonzero(1 - x):
                    candidate = x.copy()
                    candidate[i], candidate[j] = 0, 1
                    trial = cost(candidate)
                    if trial < best_value - 1e-12:
                        best, best_value = candidate, trial
            if best_value >= value - 1e-12:
                return x
            x, value = best, best_value

    def record(name: str, candidates: list[np.ndarray], refine: bool = False) -> None:
        nonlocal evaluations
        evaluations = 0
        begin = time.perf_counter()
        candidates = [swaps(x) for x in candidates] if refine else candidates
        values = [cost(x) for x in candidates]
        winner = int(np.argmin(values))
        results[name] = {"true_cost": values[winner],
                         "indices": np.flatnonzero(candidates[winner]).tolist(),
                         "objective_evaluations": evaluations,
                         "seconds": time.perf_counter() - begin}

    random = []
    for _ in range(samples):
        x = np.zeros(n, dtype=int)
        x[rng.choice(n, k, replace=False)] = 1
        random.append(x)
    record("random_feasible", random)
    record("random_plus_swaps", random[:starts], refine=True)
    record("rounded_plus_swaps", [rounded], refine=True)
    evaluations = 0
    begin = time.perf_counter()
    x = np.zeros(n, dtype=int)
    for _ in range(k):
        candidates = []
        for j in np.flatnonzero(1 - x):
            candidate = x.copy()
            candidate[j] = 1
            candidates.append(candidate)
        x = candidates[int(np.argmin([cost(c) for c in candidates]))]
    results["greedy"] = {"true_cost": cost(x), "indices": np.flatnonzero(x).tolist(),
                         "objective_evaluations": evaluations,
                         "seconds": time.perf_counter() - begin}
    return results
