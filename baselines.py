"""Section-9 baselines and diagnostics. All costs come from the OFFICIAL scorer (official/scorer.py).

Enumeration of the 256 schedules is used here only as a benchmark (C*, C_rand, elite set); it never feeds the
QUBO or the QAOA training.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "official"))
import scorer                                            # noqa: E402  (official evaluator, unmodified)
from decimal import Decimal, ROUND_HALF_UP               # noqa: E402


def official_cost(sc, bits):
    """bits: 8-char string or list of 8 ints."""
    return scorer.evaluate(sc, bits)["cost"]


def enumerate_all(sc):
    """Official cost of every schedule string -> dict (benchmark only)."""
    return {"".join(map(str, b)): official_cost(sc, list(b)) for b in product([0, 1], repeat=8)}


def elite_set(all_costs, k=13):
    """The k lowest-cost schedules, ties broken by ascending bitstring."""
    return sorted(all_costs, key=lambda s: (all_costs[s], s))[:k]


def sampling_quality(counts, all_costs):
    """E_Q, lift, elite enrichment of a counts dict (section 8)."""
    shots = sum(counts.values())
    e_q = sum(all_costs[s] * n for s, n in counts.items()) / shots
    c_star, c_rand = min(all_costs.values()), float(np.mean(list(all_costs.values())))
    elite = elite_set(all_costs)
    frac = sum(counts.get(s, 0) for s in elite) / shots
    lift = (c_rand - e_q) / (c_rand - c_star) if c_rand > c_star else float("nan")
    return {"E_Q": e_q, "C_star": c_star, "C_rand": c_rand, "lift": lift,
            "lift_clipped": min(1.0, max(0.0, lift)), "elite_fraction": frac,
            "elite_enrichment": frac / (len(elite) / 256), "elite_set": elite}


def best_alternating(sc):
    """Baseline 1: best of the four alternating schedules (A in {1010,0101}) x (B in {1010,0101})."""
    cands = [a + b for a in ("1010", "0101") for b in ("1010", "0101")]
    best = min(cands, key=lambda s: (official_cost(sc, s), s))
    return best, official_cost(sc, best)


def greedy_schedule(sc):
    """Baseline 2: per-slot greedy. At each slot every intersection serves the approach with the larger
    queue after arrivals (ties -> NS); simulated forward with the A->B transfers of the traffic model."""
    cap, link = sc["service_capacity"], sc["link"]
    queue = {(n, ph): 0 for n in "AB" for ph in ("NS", "EW")}
    pending = [0] * 4
    bits = {"A": [], "B": []}
    for t in range(4):
        avail = {}
        for n in "AB":
            for ph in ("NS", "EW"):
                inflow = sc["arrivals"][n][ph][t] + (pending[t] if (n == "B" and ph == link["phase"]) else 0)
                avail[(n, ph)] = queue[(n, ph)] + inflow
        for n in "AB":
            green = "NS" if avail[(n, "NS")] >= avail[(n, "EW")] else "EW"
            bits[n].append(1 if green == "NS" else 0)
            for ph in ("NS", "EW"):
                dep = min(avail[(n, ph)], cap) if ph == green else 0
                queue[(n, ph)] = avail[(n, ph)] - dep
                if n == "A" and ph == link["phase"] and t < 3:
                    amt = Decimal(str(link["fraction"])) * Decimal(dep)
                    pending[t + 1] += int(amt.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    s = "".join(map(str, bits["A"] + bits["B"]))
    return s, official_cost(sc, s)


def qubo_energy_str(Q, offset, s):
    x = np.array([int(c) for c in s], float)
    return float(offset + x @ np.triu(Q) @ x)


def qubo_minimiser(Q, offset):
    """Baseline 5: exact minimiser of OUR QUBO (256 evaluations of the QUBO, not of the traffic cost)."""
    return min(("".join(map(str, b)) for b in product([0, 1], repeat=8)),
               key=lambda s: (qubo_energy_str(Q, offset, s), s))


def greedy_bit_flip(Q, offset, s):
    """Hybrid refinement (NOT raw QAOA): repeat the best single-bit flip while it lowers the QUBO energy."""
    cur = s
    while True:
        flips = [cur[:i] + ("1" if cur[i] == "0" else "0") + cur[i + 1:] for i in range(8)]
        best = min(flips, key=lambda f: (qubo_energy_str(Q, offset, f), f))
        if qubo_energy_str(Q, offset, best) < qubo_energy_str(Q, offset, cur) - 1e-12:
            cur = best
        else:
            return cur
