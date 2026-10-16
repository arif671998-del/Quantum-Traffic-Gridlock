"""Collects every number quoted in README.md / REPORT.md from files in outputs/ and logs/, plus two fresh checks.

    python report_numbers.py      -> logs/verification.json, logs/report_numbers.json (and a printout)

Fresh checks (written to logs/verification.json):
  * our queue recursion (traffic_qubo.official_cost) equals official/scorer.py on all 256 schedules of R and of 30
    random scenarios (scenario_gen, seed 2026) = 31 x 256 = 7,936 evaluations;
  * the QUBO equals the global least-squares quadratic fit of the official cost (max abs difference over the 31 scenarios).
Circuit statistics (gate counts, depth) are read from the circuit rebuilt from outputs/R/seed0.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np

import baselines as bl
import qaoa
from scenario_gen import random_scenario
from traffic_qubo import build_qubo, official_cost, qubo_energy

ROOT = Path(__file__).resolve().parent
ALL = [list(b) for b in product([0, 1], repeat=8)]
X = np.array(ALL, float)
COLS = np.stack([np.ones(256)] + [X[:, i] for i in range(8)] + [X[:, i] * X[:, j] for i in range(8) for j in range(i + 1, 8)], 1)


def load(p):
    return json.loads((ROOT / p).read_text())


def verification():
    rng = np.random.default_rng(2026)
    scs = [load("official/scenario_R.json")] + [random_scenario(rng) for _ in range(30)]
    n_eval, mismatches, max_fit = 0, 0, 0.0
    for sc in scs:
        y = np.array([bl.official_cost(sc, x) for x in ALL])
        mine = np.array([official_cost(sc, x) for x in ALL])
        n_eval += 256
        mismatches += int((np.abs(mine - y) > 1e-9).sum())
        Q, off = build_qubo(sc)
        coef, *_ = np.linalg.lstsq(COLS, y, rcond=None)
        e = np.array([qubo_energy(Q, off, x) for x in ALL])
        max_fit = max(max_fit, float(np.abs(e - COLS @ coef).max()))
    return {"scenarios": len(scs), "evaluations": n_eval, "recursion_vs_scorer_mismatches": mismatches,
            "max_abs_diff_qubo_vs_global_quadratic_fit": max_fit}


def main():
    ver = verification()
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "verification.json").write_text(json.dumps(ver, indent=2))

    def seeds(name):
        return [load(f"outputs/{name}/seed{s}/metrics.json") for s in range(5)]

    def table(ms):
        return [{"seed": m["seed"], "E_Q": m["E_Q"], "lift": m["lift"], "enrichment": m["elite_enrichment"],
                 "runtime_seconds": m["runtime_seconds"], "raw_best_sample": m["baselines"]["qaoa_raw_best_sample"],
                 "most_frequent": m["baselines"]["qaoa_raw_most_frequent"], "hybrid": m["baselines"]["hybrid_refined (NOT raw QAOA)"],
                 "evals": m["qaoa"]["training_evaluations"]} for m in ms]

    R = seeds("R")
    out = {"verification": ver, "R": {"per_seed": table(R), "min_lift": min(m["lift"] for m in R),
                                      "mean_lift": float(np.mean([m["lift"] for m in R])),
                                      "baselines_seed0": R[0]["baselines"], "C_star": R[0]["C_star"], "C_rand": R[0]["C_rand"],
                                      "qubo_cost_correlation": R[0]["qubo_cost_correlation"],
                                      "gammas": R[0]["qaoa"]["gammas"], "betas": R[0]["qaoa"]["betas"],
                                      "initial_objective": R[0]["qaoa"]["initial_objective"], "final_objective": R[0]["qaoa"]["final_objective"]}}
    for n in (1, 2):
        ms = seeds(f"heldout_{n}")
        out[f"heldout_{n}"] = {"per_seed": table(ms), "min_lift": min(m["lift"] for m in ms), "C_star": ms[0]["C_star"], "C_rand": ms[0]["C_rand"],
                               "baselines_seed0": ms[0]["baselines"]}
    # circuit statistics of the file actually written for R seed 0
    q = load("outputs/R/seed0/qubo.json")
    m0 = R[0]
    qc = qaoa.add_measurements(qaoa.bound_circuit(np.array(q["ising"]["h"]), np.array(q["ising"]["J"]), q["ising"]["sigma"],
                                                  m0["qaoa"]["gammas"], m0["qaoa"]["betas"]))
    ops = {k: int(v) for k, v in qc.count_ops().items()}
    out["circuit_R_seed0"] = {"count_ops": ops, "depth": qc.depth(), "num_qubits": qc.num_qubits,
                              "coupled_pairs": int(np.count_nonzero(np.triu(np.array(q["ising"]["J"]), 1)))}
    g = load("logs/generalisation.json")
    worst = sorted(g["scenarios"], key=lambda r: -r["qubo_regret_pct"])[:5]
    out["generalisation"] = {"summary": g["summary"], "worst_regret": [{k: r[k] for k in ("name", "qubo_regret_pct", "lift", "enrichment")} for r in worst]}
    a = load("logs/ablations.json")
    out["ablations"] = {"decision": a["decision"], "summary": {n: {"group": s["group"], **s["random30"], "R_lift": s["R"]["lift"], "R_enrichment": s["R"]["enrichment"]}
                                                                for n, s in a["summary"].items()}}
    out["versions"] = load("logs/versions.json")
    (ROOT / "logs" / "report_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("verification", "circuit_R_seed0")}, indent=1))
    print(json.dumps(out["R"], indent=1))
    print(json.dumps({k: out[k] for k in ("heldout_1", "heldout_2")}, indent=1)[:3500])
    print(json.dumps(out["generalisation"], indent=1))


if __name__ == "__main__":
    main()
