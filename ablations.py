"""Ablation study: depth, initialisation, surrogate degree, objective.  Writes logs/ablations.json and .md.

    python ablations.py

Every configuration is run on R and on 30 random scenarios (scenario_gen, seed 2026). Settings are chosen ONLY from the
30 random scenarios (never from R or H). Each run: train (<= 300 evaluations), bind, sample 4,096 shots with seed 0,
score with the official scorer.  Scenarios with C_rand - C* < 1.0 are excluded from lift statistics (ratio unstable).

Decision rule 1 (fixed before the first run): change the initial-default ramp scale (0.6, 0.6) only if another ramp-scale setting
raises the mean lift on the 30 random scenarios by >= 0.02 WITHOUT lowering the minimum lift; otherwise keep it.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import baselines as bl
import qaoa
from scenario_gen import random_scenario
from traffic_qubo import build_qubo, to_ising

ROOT = Path(__file__).resolve().parent
MIN_DENOM = 1.0
GAMMA_SCALES, BETA_SCALES = [0.4, 0.6, 0.8, 1.0], [0.4, 0.6, 0.8]


def configs():
    """name -> dict(group, p, init, degree, cvar). init = ('ramp', gamma_scale, beta_scale) or ('random', seed)."""
    c = {"baseline (p=4, ramp 0.6/0.6, degree 2, mean)": dict(group="baseline", p=4, init=("ramp", 0.6, 0.6), degree=2, cvar=None)}
    for p in (1, 2, 3):
        c[f"depth p={p}"] = dict(group="depth", p=p, init=("ramp", 0.6, 0.6), degree=2, cvar=None)
    for s in (1, 2, 3):
        c[f"init random angles U(0,1), seed {s}"] = dict(group="init", p=4, init=("random", s), degree=2, cvar=None)
    for g in GAMMA_SCALES:
        for b in BETA_SCALES:
            if (g, b) != (0.6, 0.6):
                c[f"init ramp gamma {g} / beta {b}"] = dict(group="scale", p=4, init=("ramp", g, b), degree=2, cvar=None)
    # Added after the first study (16 Oct, ~17:45): same magnitudes as the adopted (1.0, 0.4) but beta negative, because
    # RX(2 beta) = exp(-i beta X) needs the opposite sign to gamma to start an annealing-like ramp on the low-energy side.
    c["init ramp gamma 1.0 / beta -0.4 (sign-corrected)"] = dict(group="sign", p=4, init=("ramp", 1.0, -0.4), degree=2, cvar=None)
    # Main ablation block on the ADOPTED ramp (1.0, -0.4) (added 16 Oct, after the sign fix). The old rows below stay as the
    # "before the sign fix" block. Random-angle starts do not depend on the ramp, so they serve both blocks unchanged.
    for p in (1, 2, 3):
        c[f"[adopted ramp] depth p={p}"] = dict(group="a_depth", p=p, init=("ramp", 1.0, -0.4), degree=2, cvar=None)
    c["[adopted ramp] surrogate degree 1 (linear only)"] = dict(group="a_surrogate", p=4, init=("ramp", 1.0, -0.4), degree=1, cvar=None)
    c["[adopted ramp] objective CVaR 0.25"] = dict(group="a_objective", p=4, init=("ramp", 1.0, -0.4), degree=2, cvar=0.25)
    c["surrogate degree 1 (linear only)"] = dict(group="surrogate", p=4, init=("ramp", 0.6, 0.6), degree=1, cvar=None)
    c["objective CVaR 0.25"] = dict(group="objective", p=4, init=("ramp", 0.6, 0.6), degree=2, cvar=0.25)
    return c


def init_angles(spec, p):
    if spec[0] == "ramp":
        return qaoa.initial_angles(p, spec[1], spec[2])
    rng = np.random.default_rng(spec[1])
    return rng.uniform(0, 1, p), rng.uniform(0, 1, p)


def run_one(sc, allc, cfg):
    Q, off = build_qubo(sc, degree=cfg["degree"])
    h, J, _ = to_ising(Q, off)
    sigma = qaoa.ising_sigma(h, J)
    qmin = bl.qubo_minimiser(Q, off)
    q = bl.sampling_quality  # alias for brevity
    tr = qaoa.train(h, J, sigma, p=cfg["p"], init=init_angles(cfg["init"], cfg["p"]), cvar_alpha=cfg["cvar"])
    counts, _ = qaoa.sample(qaoa.bound_circuit(h, J, sigma, tr["gammas"], tr["betas"]), 0)
    m = q(counts, allc)
    return {"lift": m["lift"], "enrichment": m["elite_enrichment"], "E_Q": m["E_Q"], "evals": tr["evaluations"],
            "initial_objective": tr["initial_objective"],
            "regret_pct": 100.0 * (allc[qmin] - m["C_star"]) / m["C_star"]}


def run_scenario(args):
    """Worker: all configurations on one scenario."""
    sc, = args
    allc = bl.enumerate_all(sc)
    denom = float(np.mean(list(allc.values()))) - min(allc.values())
    return {"name": sc["name"], "denominator": denom,
            "results": {name: run_one(sc, allc, cfg) for name, cfg in configs().items()}}


def summarise(rows, name):
    ok = [r["results"][name] for r in rows if r["denominator"] >= MIN_DENOM]
    lifts = np.array([r["lift"] for r in ok])
    reg = np.array([r["regret_pct"] for r in ok])
    return {"n": len(ok), "lift_mean": float(lifts.mean()), "lift_min": float(lifts.min()),
            "enrichment_mean": float(np.mean([r["enrichment"] for r in ok])),
            "regret_mean_pct": float(reg.mean()), "regret_share_le_5pct": float((reg <= 5).mean()),
            "initial_objective_mean": float(np.mean([r["initial_objective"] for r in ok])),
            "evals_max": int(max(r["evals"] for r in ok))}


def main():
    rng = np.random.default_rng(2026)
    rand = [random_scenario(rng, f"rand{i:02d}") for i in range(30)]
    R = json.loads((ROOT / "official" / "scenario_R.json").read_text())
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(run_scenario, [(s,) for s in rand + [R]]))
    r_row, rand_rows = rows[-1], rows[:-1]
    cfgs = configs()
    summary = {n: {"group": c["group"], "random30": summarise(rand_rows, n),
                   "R": r_row["results"][n]} for n, c in cfgs.items()}

    # ---- decision rule (random scenarios only) ----
    base = summary["baseline (p=4, ramp 0.6/0.6, degree 2, mean)"]["random30"]
    qualifying = {n: s["random30"] for n, s in summary.items() if s["group"] == "scale"
                  and s["random30"]["lift_mean"] >= base["lift_mean"] + 0.02 and s["random30"]["lift_min"] >= base["lift_min"]}
    best = max(qualifying, key=lambda n: qualifying[n]["lift_mean"]) if qualifying else None
    decision = {"rule": "change ramp scale only if mean lift (30 random scenarios) rises >= 0.02 and min lift does not fall",
                "baseline_mean_lift": base["lift_mean"], "baseline_min_lift": base["lift_min"],
                "qualifying": list(qualifying), "chosen": best or "keep 0.6/0.6"}

    # ---- second decision (same rule), sign-corrected ramp versus the adopted (1.0, +0.4), random scenarios only ----
    cur = summary["init ramp gamma 1.0 / beta 0.4"]["random30"]
    new = summary["init ramp gamma 1.0 / beta -0.4 (sign-corrected)"]["random30"]
    passes = new["lift_mean"] >= cur["lift_mean"] + 0.02 and new["lift_min"] >= cur["lift_min"]
    decision["second_rule_sign_correction"] = {
        "rule": "same rule: adopt (1.0, -0.4) only if mean lift rises >= 0.02 over the current default (1.0, 0.4) and min lift does not fall",
        "current_default_mean_lift": cur["lift_mean"], "current_default_min_lift": cur["lift_min"],
        "candidate_mean_lift": new["lift_mean"], "candidate_min_lift": new["lift_min"], "passes": bool(passes)}

    out = {"random_scenarios": {"generator_seed": 2026, "count": 30,
                                "excluded_for_small_denominator": [r["name"] for r in rand_rows if r["denominator"] < MIN_DENOM]},
           "summary": summary, "decision": decision, "per_scenario": rows}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "ablations.json").write_text(json.dumps(out, indent=1))

    # ---- markdown tables ----
    L = ["# Ablations (generated by ablations.py)", "",
         f"30 random scenarios (seed 2026; {out['random_scenarios']['excluded_for_small_denominator'] or 'none'} excluded for "
         f"C_rand - C* < {MIN_DENOM}) + scenario R. Lift/enrichment from 4,096 sampled shots, seed 0. Regret = (official cost of "
         "the QUBO minimiser - C*) / C* in %.", ""]
    head = "| setting | mean lift | min lift | mean enrichment | mean regret % | share regret <= 5% | max evals | mean initial objective | R lift | R enrichment |\n|---|---|---|---|---|---|---|---|---|---|"
    ADOPT = "init ramp gamma 1.0 / beta -0.4 (sign-corrected)"
    OLDDEF = "init ramp gamma 1.0 / beta 0.4"

    def block(title, names, level="##"):
        L.extend([f"{level} {title}", "", head])
        for n in names:
            a, r = summary[n]["random30"], summary[n]["R"]
            L.append(f"| {n} | {a['lift_mean']:.3f} | {a['lift_min']:.3f} | {a['enrichment_mean']:.2f} | {a['regret_mean_pct']:.2f} | "
                     f"{a['regret_share_le_5pct']:.2f} | {a['evals_max']} | {a['initial_objective_mean']:.2f} | {r['lift']:.3f} | {r['enrichment']:.2f} |")
        L.append("")

    def grp(*g):
        return [n for n, s in summary.items() if s["group"] in g]

    L += ["# Main ablations: adopted ramp (gamma 1.0, beta -0.4)", ""]
    block("Depth", [ADOPT] + grp("a_depth"), "###")
    block("Initialisation: adopted ramp vs random angles", [ADOPT] + grp("init"), "###")
    block("Surrogate degree", [ADOPT] + grp("a_surrogate"), "###")
    block("Objective", [ADOPT] + grp("a_objective"), "###")
    L += ["# Before the sign fix (beta of the same sign as gamma: starts ABOVE the random energy level)", ""]
    block("Old default and old initial setting", grp("baseline") + [OLDDEF], "###")
    block("Old depth rows (ramp 0.6/0.6)", grp("baseline", "depth"), "###")
    block("Old ramp-scale grid", grp("baseline", "scale"), "###")
    block("Old surrogate / objective rows (ramp 0.6/0.6)", grp("baseline", "surrogate", "objective"), "###")
    L += ["## Decision", "", f"Rule: {decision['rule']}.", f"Baseline (initial default): mean {decision['baseline_mean_lift']:.3f}, min {decision['baseline_min_lift']:.3f}.",
          f"Qualifying settings: {decision['qualifying'] or 'none'}.  **Chosen: {decision['chosen']}.**", "",
          "### Second decision: sign correction", "", f"Rule: {decision['second_rule_sign_correction']['rule']}.",
          f"Current default (1.0, 0.4): mean {cur['lift_mean']:.3f}, min {cur['lift_min']:.3f}. Candidate (1.0, -0.4): mean {new['lift_mean']:.3f}, "
          f"min {new['lift_min']:.3f}. **Passes: {passes}.**", ""]
    (ROOT / "logs" / "ablations.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
