import json, numpy as np
from scipy.optimize import minimize
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from proto import cost_table, qubo_walsh2, qaoa_probs, metrics, rand_scenario, elite_mask

def train(E, p, init=(0.6,0.6), obj="mean", alpha=0.2, maxeval=300, starts=1):
    En = (E - E.mean()) / E.std()
    order = np.argsort(En)
    def f(x):
        pr = qaoa_probs(En, x[:p], x[p:])
        if obj == "mean":
            return float(pr @ En)
        # CVaR_alpha of energy distribution
        ps = pr[order]; cs = np.cumsum(ps); m = cs <= alpha
        k = np.searchsorted(cs, alpha)
        w = ps.copy(); w[k+1:] = 0; w[k] = alpha - (cs[k-1] if k > 0 else 0)
        return float(w @ En[order]) / alpha
    k = (np.arange(p) + 0.5) / p
    best = None
    inits = [init] if starts == 1 else [(0.4,0.4),(0.6,0.6),(0.9,0.5)][:starts]
    for dg, db in inits:
        x0 = np.concatenate([dg * k, db * (1 - k)])
        r = minimize(f, x0, method="COBYLA", options={"maxiter": maxeval // len(inits), "rhobeg": 0.2})
        if best is None or r.fun < best.fun: best = r
    return best.x, En

scs = [json.load(open(os.path.join(os.path.dirname(__file__), "..", "official", "scenario_R.json")))]
rng = np.random.default_rng(1)
scs += [rand_scenario(rng) for _ in range(20)]
rng2 = np.random.default_rng(7)
scs += [rand_scenario(rng2) for _ in range(19)]
tables = [cost_table(s) for s in scs]
Qs = [qubo_walsh2(C)[0] for C in tables]
variants = {
  "p4 mean ramp.6": dict(p=4),
  "p4 mean 3starts": dict(p=4, starts=3),
  "p4 cvar.25": dict(p=4, obj="cvar", alpha=0.25),
  "p4 cvar.1": dict(p=4, obj="cvar", alpha=0.1),
  "p3 mean": dict(p=3),
}
for name, kw in variants.items():
    L, Enr = [], []
    for C, Q in zip(tables, Qs):
        p = kw["p"]
        x, En = train(Q, **kw)
        pr = qaoa_probs(En, x[:p], x[p:])
        l, e = metrics(C, pr); L.append(l); Enr.append(e)
    L, Enr = np.array(L), np.array(Enr)
    print(f"{name:18s} lift mean {L.mean():.3f} min {L.min():.2f} p10 {np.percentile(L,10):.2f} | enrich mean {Enr.mean():.1f} min {Enr.min():.1f} | R: {L[0]:.2f}/{Enr[0]:.1f}", flush=True)
