"""Quantum Traffic Gridlock pipeline.

    python run.py --scenario official/scenario_R.json --seed 0 --out outputs/R/seed0

Steps: read scenario -> build QUBO (traffic_qubo.py) -> Ising form -> p=4 QAOA circuit -> train angles with COBYLA on
the exact statevector expectation (<= 300 evaluations) -> bind angles -> sample 4,096 shots with a seeded sampler ->
write qubo.json, circuit.qasm, counts.json, schedule.json (+ metrics.json with baselines and diagnostics).
Nothing here depends on the scenario being R; the official scorer is used only AFTER sampling to report costs.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import qiskit
from qiskit import qasm3

import baselines as bl
import qaoa
from traffic_qubo import build_qubo, to_ising

SHOTS = 4096
BIT_ORDER = ("Eight characters: A slot 1, A slot 2, A slot 3, A slot 4, B slot 1, B slot 2, B slot 3, B slot 4 "
             "(left to right). 1 = North-South green, 0 = East-West green.")
NAMES = ["A slot 1", "A slot 2", "A slot 3", "A slot 4", "B slot 1", "B slot 2", "B slot 3", "B slot 4"]


def bitstring_to_schedule(s):
    """Printed Qiskit bitstring (clbit 7 first) -> list of schedule bits [A1..A4, B1..B4]."""
    return [int(c) for c in s]


def schedule_to_bitstring(bits):
    return "".join(str(int(b)) for b in bits)


def _write(out, name, obj):
    (out / name).write_text(json.dumps(obj, indent=2))


def run_pipeline(scenario_path, seed, out_dir):
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sc = json.loads(Path(scenario_path).read_text())
    bl.scorer.validate_scenario(sc)

    # ---- 1. QUBO and Ising form (built from the traffic recursion only) ------------------------------------
    Q, offset = build_qubo(sc)
    h, J, c = to_ising(Q, offset)
    sigma = qaoa.ising_sigma(h, J)
    _write(out, "qubo.json", {
        "scenario": sc.get("name"),
        "convention": "energy(x) = offset + sum_i Q[i][i] x_i + sum_{i<j} Q[i][j] x_i x_j, x_i in {0,1}; "
                      "Ising: x_i = (1 - z_i)/2, energy = c + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j (z=+1 <-> bit 0)",
        "Q": Q.tolist(), "offset": offset,
        "ising": {"h": h.tolist(), "J": J.tolist(), "offset": c, "sigma": sigma},
        "variables": [{"variable": f"x{j}", "name": NAMES[j], "schedule_bit": j,
                       "qubit": qaoa.qubit_of(j), "clbit": qaoa.qubit_of(j)} for j in range(8)],
        "auxiliary_variables": [],
        "bit_order": BIT_ORDER,
    })

    # ---- 2. Train the angles (exact statevector, no shots, independent of the seed) ------------------------
    tr = qaoa.train(h, J, sigma)
    circuit = qaoa.bound_circuit(h, J, sigma, tr["gammas"], tr["betas"])

    # ---- 3. Final sampling: 4,096 shots from the bound circuit ----------------------------------------------
    counts, measured = qaoa.sample(circuit, seed, SHOTS)
    assert sum(counts.values()) == SHOTS
    _write(out, "counts.json", counts)
    (out / "circuit.qasm").write_text(qasm3.dumps(measured))

    # ---- 4. Decode schedules (raw QAOA first, hybrid separately labelled) -----------------------------------
    qe = {s: bl.qubo_energy_str(Q, offset, s) for s in counts}
    raw_best = min(counts, key=lambda s: (qe[s], s))                 # lowest QUBO energy among the samples
    most_freq = min(counts, key=lambda s: (-counts[s], s))
    hybrid = bl.greedy_bit_flip(Q, offset, raw_best)

    def entry(s, label, **extra):
        r = bl.scorer.evaluate(sc, s)
        return {"label": label, "bits": s, "A": bitstring_to_schedule(s)[:4], "B": bitstring_to_schedule(s)[4:],
                "official_cost": r["cost"], "qubo_energy": bl.qubo_energy_str(Q, offset, s),
                "shots": counts.get(s, 0), **extra}

    _write(out, "schedule.json", {
        "bit_order": BIT_ORDER,
        "primary": entry(raw_best, "raw QAOA: sampled schedule with the lowest QUBO energy"),
        "most_frequent": entry(most_freq, "raw QAOA: most frequent sample"),
        "hybrid": entry(hybrid, "hybrid (classical refinement, NOT raw QAOA): greedy 1-bit-flip descent on the QUBO "
                                "starting from the primary schedule"),
    })

    # ---- 5. Metrics and section-9 baselines (official scorer; enumeration is a benchmark only) ---------------
    allc = bl.enumerate_all(sc)
    q = bl.sampling_quality(counts, allc)
    alt, greedy = bl.best_alternating(sc), bl.greedy_schedule(sc)
    qmin = bl.qubo_minimiser(Q, offset)
    ordered = sorted(allc)
    qen = np.array([bl.qubo_energy_str(Q, offset, s) for s in ordered])
    cst = np.array([allc[s] for s in ordered])
    best_s = min(allc, key=lambda s: (allc[s], s))
    pct = lambda cost: 100.0 * (cost - q["C_star"]) / (q["C_rand"] - q["C_star"]) if q["C_rand"] > q["C_star"] else None
    runtime = time.perf_counter() - t0
    metrics = {
        "scenario": sc.get("name"), "seed": seed, "shots": SHOTS,
        "C_star": q["C_star"], "C_star_schedule": best_s, "C_rand": q["C_rand"],
        "E_Q": q["E_Q"], "lift": q["lift"], "lift_clipped": q["lift_clipped"],
        "elite_fraction": q["elite_fraction"], "elite_enrichment": q["elite_enrichment"], "elite_set": q["elite_set"],
        "baselines": {
            "alternating_best": {"bits": alt[0], "cost": alt[1]},
            "greedy_per_slot": {"bits": greedy[0], "cost": greedy[1]},
            "enumeration_optimum": {"bits": best_s, "cost": q["C_star"]},
            "uniform_random_mean": q["C_rand"],
            "qubo_minimiser": {"bits": qmin, "cost": allc[qmin],
                               "regret_percent_of_C_star": 100.0 * (allc[qmin] - q["C_star"]) / q["C_star"],
                               "regret_percent_of_range": pct(allc[qmin])},
            "qaoa_raw_best_sample": {"bits": raw_best, "cost": allc[raw_best]},
            "qaoa_raw_most_frequent": {"bits": most_freq, "cost": allc[most_freq]},
            "hybrid_refined (NOT raw QAOA)": {"bits": hybrid, "cost": allc[hybrid]},
        },
        "qubo_cost_correlation": float(np.corrcoef(qen, cst)[0, 1]),
        "qaoa": {"p": qaoa.P, "n_qubits": qaoa.N, "optimizer": "COBYLA (scipy), maxiter 300, rhobeg 0.2",
                 "objective": "exact statevector expectation of the sigma-normalised cost Hamiltonian",
                 "training_evaluations": tr["evaluations"], "initial_objective": tr["initial_objective"],
                 "final_objective": tr["final_objective"], "gammas": tr["gammas"], "betas": tr["betas"],
                 "sigma": sigma},
        "runtime_seconds": runtime,
        "versions": {"python": platform.python_version(), "qiskit": qiskit.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
    }
    metrics["runtime_seconds"] = time.perf_counter() - t0
    _write(out, "metrics.json", metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = run_pipeline(a.scenario, a.seed, a.out)
    print(f"scenario={m['scenario']} seed={a.seed} lift={m['lift']:.4f} enrichment={m['elite_enrichment']:.2f} "
          f"E_Q={m['E_Q']:.2f} C*={m['C_star']} C_rand={m['C_rand']:.2f} evals={m['qaoa']['training_evaluations']} "
          f"runtime={m['runtime_seconds']:.1f}s -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
