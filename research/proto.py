"""Prototype (numpy only): QUBO surrogate quality and QAOA sampling quality for PS2.
Not the submission; used to choose the approach before implementation."""
import json, sys, time
import numpy as np
from scipy.optimize import minimize
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "official")); from scorer import evaluate

N = 8
IDX = np.arange(256)
BITS = ((IDX[:, None] >> (7 - np.arange(8))[None, :]) & 1).astype(float)   # BITS[i, j] = schedule bit j of state i
STR = [format(i, "08b") for i in range(256)]


def cost_table(sc):
    return np.array([evaluate(sc, s)["cost"] for s in STR])


def quad_design():
    cols = [np.ones(256)] + [BITS[:, j] for j in range(8)]
    pairs = [(j, k) for j in range(8) for k in range(j + 1, 8)]
    cols += [BITS[:, j] * BITS[:, k] for j, k in pairs]
    return np.stack(cols, 1), pairs


X2, PAIRS = quad_design()


def qubo_walsh2(C):
    coef, *_ = np.linalg.lstsq(X2, C, rcond=None)
    return X2 @ coef, coef


def elite_mask(C):
    order = sorted(range(256), key=lambda i: (C[i], STR[i]))
    m = np.zeros(256, bool); m[order[:13]] = True
    return m


def rx_all(psi, beta):
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * N)
    for q in range(N):
        psi = np.moveaxis(psi, q, 0)
        a, b = psi[0].copy(), psi[1].copy()
        psi[0] = c * a + s * b; psi[1] = s * a + c * b
        psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)


def qaoa_probs(E, gam, bet):
    psi = np.full(256, 1 / 16, complex)
    for g, b in zip(gam, bet):
        psi = psi * np.exp(-1j * g * E)
        psi = rx_all(psi, b)
    return np.abs(psi) ** 2


def train(E, p, maxeval=300):
    En = (E - E.mean()) / E.std()
    nev = [0]
    def f(x):
        nev[0] += 1
        pr = qaoa_probs(En, x[:p], x[p:])
        return float(pr @ En)
    k = (np.arange(p) + 0.5) / p
    best = None
    budget = maxeval
    for dg, db in [(0.6, 0.6)]:
        x0 = np.concatenate([dg * k, db * (1 - k)])
        r = minimize(f, x0, method="COBYLA", options={"maxiter": budget, "rhobeg": 0.2})
        if best is None or r.fun < best.fun:
            best = r
    return best.x, nev[0], En


def metrics(C, pr):
    Cstar, Crand = C.min(), C.mean()
    EQ = pr @ C
    lift = (Crand - EQ) / (Crand - Cstar)
    enr = pr[elite_mask(C)].sum() / (13 / 256)
    return lift, enr


def rand_scenario(rng):
    return {"arrivals": {n: {ph: [int(v) for v in rng.integers(0, 9, 4)] for ph in ("NS", "EW")} for n in "AB"},
            "service_capacity": int(rng.integers(4, 7)), "storage_capacity": int(rng.integers(5, 9)),
            "overflow_weight": float(rng.choice([2, 3, 4, 5, 6])), "switch_weight": float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])),
            "link": {"from": "A", "to": "B", "phase": str(rng.choice(["NS", "EW"])), "fraction": float(rng.choice([0.4, 0.5, 0.6, 0.7, 0.8]))}}


if __name__ == "__main__":
    scs = [json.load(open(os.path.join(os.path.dirname(__file__), "..", "official", "scenario_R.json")))]
    rng = np.random.default_rng(1)
    scs += [rand_scenario(rng) for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 20)]
    rows = []
    for si, sc in enumerate(scs):
        C = cost_table(sc)
        Q, _ = qubo_walsh2(C)
        arg = int(np.argmin(Q))
        regret = C[arg] / C.min() - 1
        corr = np.corrcoef(Q, C)[0, 1]
        res = [si, round(C.min(), 1), round(C.mean(), 1), round(regret * 100, 1), round(corr, 3)]
        for p in (1, 2, 3, 4):
            x, nev, En = train(Q, p)
            pr = qaoa_probs(En, x[:p], x[p:])
            lift, enr = metrics(C, pr)
            res += [round(lift, 2), round(enr, 1)]
        rows.append(res)
        print(res, flush=True)
    A = np.array(rows, float)
    print("mean regret% {:.1f}  max {:.1f}  share<=5% {:.2f}  mean corr {:.3f}".format(A[:, 3].mean(), A[:, 3].max(), (A[:, 3] <= 5).mean(), A[:, 4].mean()))
    for i, p in enumerate((1, 2, 3, 4)):
        print(f"p={p}: lift mean {A[:, 5 + 2*i].mean():.2f} min {A[:, 5 + 2*i].min():.2f} | enrich mean {A[:, 6 + 2*i].mean():.1f} min {A[:, 6 + 2*i].min():.1f}")
