"""Traffic model and QUBO construction for the Quantum Traffic Gridlock challenge.

Decision bits: x[0..3] = intersection A, slots 1..4; x[4..7] = intersection B, slots 1..4.
x = 1 means North-South green, x = 0 means East-West green (problem statement section 3).

Modelling rule (scenario-independent):
  The official cost is a sum of local terms, one per (intersection, approach, slot):
      term = queue_after_service + overflow_weight * max(0, queue_after_service - storage)
  plus the switching cost. Each local term depends only on the bits in its causal window:
      approach at A, slot t           -> A bits 1..t
      unlinked approach at B, slot t  -> B bits 1..t
      linked approach at B, slot t    -> B bits 1..t and A bits 1..t-1 (vehicles transferred over the link)
  We evaluate each term with our own implementation of the queue recursion over its window and
  replace it by its best quadratic approximation (degree-2 truncation of its Walsh-Fourier
  expansion = least squares over the window's configurations with basis 1, x_i, x_i*x_j).
  The switching cost w * sum [x_t != x_(t-1)] = w * sum (x_t + x_(t-1) - 2 x_t x_(t-1)) is already
  quadratic and is added exactly. No auxiliary variables are needed (8 variables).
  Terms that are at most quadratic are reproduced exactly; only higher-order interactions created
  by the min(capacity, queue) and max(0, queue - storage) nonlinearities are approximated.
  The global optimum is never computed or used to build the QUBO.
"""
from decimal import Decimal, ROUND_HALF_UP
from itertools import combinations
import numpy as np

NODES, PHASES = ("A", "B"), ("NS", "EW")


def bit_index(node, slot):
    """Schedule bit index for intersection node ('A'/'B') and slot 0..3."""
    return (0 if node == "A" else 4) + slot


def simulate(sc, x):
    """Section 3 queue recursion for one schedule x (8 ints). Returns the per-term queues after
    service, q[(node, phase, t)], and the number of switches."""
    cap, link = sc["service_capacity"], sc["link"]
    queue = {(n, ph): 0 for n in NODES for ph in PHASES}
    pending = [0, 0, 0, 0]
    q = {}
    for t in range(4):
        for n in NODES:
            for ph in PHASES:
                inflow = sc["arrivals"][n][ph][t]
                if n == "B" and ph == link["phase"]:
                    inflow += pending[t]
                avail = queue[(n, ph)] + inflow
                green = x[bit_index(n, t)] == (1 if ph == "NS" else 0)
                dep = min(avail, cap) if green else 0
                queue[(n, ph)] = avail - dep
                q[(n, ph, t)] = queue[(n, ph)]
                if n == "A" and ph == link["phase"] and t < 3:
                    amt = Decimal(str(link["fraction"])) * Decimal(dep)
                    pending[t + 1] += int(amt.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    switches = sum(x[bit_index(n, t)] != x[bit_index(n, t - 1)] for n in NODES for t in range(1, 4))
    return q, switches


def official_cost(sc, x):
    """Official cost from our own recursion (checked against scorer.py in the tests)."""
    q, sw = simulate(sc, x)
    S = sc["storage_capacity"]
    delay = sum(q.values())
    overflow = sum(max(0, v - S) for v in q.values())
    return delay + sc["overflow_weight"] * overflow + sc["switch_weight"] * sw


def causal_window(sc, node, phase, t):
    w = [bit_index(node, s) for s in range(t + 1)]
    if node == "B" and phase == sc["link"]["phase"]:
        w += [bit_index("A", s) for s in range(t)]
    return sorted(w)


def term_value(sc, node, phase, t, x):
    q, _ = simulate(sc, x)
    v = q[(node, phase, t)]
    return v + sc["overflow_weight"] * max(0, v - sc["storage_capacity"])


def quadratic_projection(values, window):
    """Least-squares fit of values (indexed by assignments of the window bits) with basis
    1, x_i, x_i x_j.  Returns (const, {i: lin}, {(i, j): quad}) in global bit indices."""
    k = len(window)
    A = np.array([[(a >> (k - 1 - r)) & 1 for r in range(k)] for a in range(2 ** k)], float)
    pairs = list(combinations(range(k), 2))
    cols = [np.ones(2 ** k)] + [A[:, r] for r in range(k)] + [A[:, r] * A[:, s] for r, s in pairs]
    coef, *_ = np.linalg.lstsq(np.stack(cols, 1), np.asarray(values, float), rcond=None)
    lin = {window[r]: coef[1 + r] for r in range(k)}
    quad = {(window[r], window[s]): coef[1 + k + i] for i, (r, s) in enumerate(pairs)}
    return coef[0], lin, quad


def build_qubo(sc, degree=2):
    """Return (Q, offset): energy(x) = offset + sum_i Q[i,i] x_i + sum_{i<j} Q[i,j] x_i x_j.
    degree=1 keeps only linear terms of each local term (ablation)."""
    Q = np.zeros((8, 8))
    offset = 0.0
    for n in NODES:
        for ph in PHASES:
            for t in range(4):
                w = causal_window(sc, n, ph, t)
                vals = []
                for a in range(2 ** len(w)):
                    x = [0] * 8
                    for r, i in enumerate(w):
                        x[i] = (a >> (len(w) - 1 - r)) & 1
                    vals.append(term_value(sc, n, ph, t, x))
                c0, lin, quad = quadratic_projection(vals, w)
                if degree == 1:      # best linear fit instead
                    k = len(w)
                    A = np.array([[(a >> (k - 1 - r)) & 1 for r in range(k)] for a in range(2 ** k)], float)
                    coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(2 ** k), A]), np.asarray(vals, float), rcond=None)
                    c0, lin, quad = coef[0], {w[r]: coef[1 + r] for r in range(k)}, {}
                offset += c0
                for i, v in lin.items():
                    Q[i, i] += v
                for (i, j), v in quad.items():
                    Q[min(i, j), max(i, j)] += v
    sw = sc["switch_weight"]                       # exact switching term
    for n in NODES:
        for t in range(1, 4):
            i, j = bit_index(n, t - 1), bit_index(n, t)
            Q[i, i] += sw; Q[j, j] += sw; Q[i, j] += -2 * sw
    return Q, offset


def qubo_energy(Q, offset, x):
    x = np.asarray(x, float)
    return float(offset + x @ np.triu(Q) @ x)    # diagonal entries act as linear terms since x_i^2 = x_i


def to_ising(Q, offset):
    """x_i = (1 - z_i)/2 with z_i = +1 for bit 0.  energy = c + sum h_i z_i + sum_{i<j} J_ij z_i z_j."""
    h = np.zeros(8); J = np.zeros((8, 8)); c = offset
    for i in range(8):
        c += Q[i, i] / 2; h[i] -= Q[i, i] / 2
        for j in range(i + 1, 8):
            v = Q[i, j]
            c += v / 4; h[i] -= v / 4; h[j] -= v / 4; J[i, j] += v / 4
    return h, J, c
