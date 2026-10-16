# AI-use disclosure — Quantum Traffic Gridlock (team: Arif, Rohit, Mehul)

Problem statement section 10 permits AI assistants provided assistance is declared and results are independently
verified. This page declares all AI assistance in this submission. The detailed, dated record is `AI_USE_LOG.md`.

## 1. Tools and what each did

| tool | what it did |
|---|---|
| **Claude (Claude app)** | Explained the problem statement and the quantum concepts to the team (all three of us are beginners in quantum computing); built the numpy prototype in `research/` that compared QUBO surrogates and QAOA training variants; wrote `traffic_qubo.py` (queue recursion, causal windows, local quadratic projection, QUBO and Ising conversion) and the project brief `CLAUDE.md`; typesets the PDF versions of the report and this disclosure. |
| **Claude Code** (model: Claude Sonnet 5.5) | Wrote the tests (`tests/`), `qaoa.py`, `baselines.py`, `run.py`, `scenario_gen.py`, `ablations.py`, `generalisation.py`, `report_numbers.py` and `make_outputs.sh`; ran them in our repository; drafted `README.md` and `REPORT.md` from the files in `outputs/` and `logs/`. |

No other AI tool was used. Libraries used (permitted): Qiskit, NumPy, SciPy.

## 2. What the team did

- Reviewed and approved the design (QUBO rule, QAOA depth, objective, optimiser, resource limits) and **approved the
  ablation selection rule before the ablation was run** (change the ramp scale only if mean lift on 30 random scenarios
  rises by ≥ 0.02 without lowering the minimum lift; R and H are never used to choose settings).
- Ran every test and experiment on our own machine and looked at the outputs.
- Checked the generated schedules with the official `scorer.py`.
- Followed the organisers' Discord announcements about scenario H (not released by submission time).
- Reviewed the report against the logs before submission.

## 3. How results were verified independently of the AI-written code

- **Official scorer:** the four reference vectors (116.0, 146.0, 237.0, 92.5) are a unit test; every cost in the report is
  computed by `official/scorer.py`; `scorer.py --schedule outputs/R/seed0/schedule.json` runs on our `schedule.json`.
- **Model check:** our queue recursion equals the scorer on 7,936 evaluations (R and 30 random scenarios × 256 schedules, 0 mismatches,
  `logs/verification.json`); the QUBO equals the global least-squares quadratic fit of the official cost (maximum
  difference 1.9e-12).
- **Tests (19, about 4 s):** bit mapping round trip, qubit order in a real circuit, `qubo.json` energies, counts sum to 4,096
  with 8-bit keys, circuit statevector vs counts (total variation within 1.5× pure shot noise), `circuit.qasm` equals the dump of the circuit
  rebuilt from the saved angles, E_Q / lift / enrichment recomputed from `counts.json` and the scorer, ≤ 300 training evaluations.
- **Determinism:** the same seed gives byte-identical `counts.json` and `circuit.qasm` (tested); re-running
  `make_outputs.sh` changes only the `runtime_seconds` field of `metrics.json`.
- **Report numbers** are produced by scripts into `outputs/` and `logs/`, not typed from memory.

## 4. Integrity statements

- The QUBO comes from a documented, scenario-independent traffic rule (local terms over causal windows, best quadratic
  approximation per term, exact switching term). It was **never** built from enumeration, from the optimal schedule or from
  any scenario-specific lookup.
- Enumeration of the 256 schedules is used only for benchmarks (C\*, C_rand, elite set, regret) after the fact; it never
  feeds the QUBO or the QAOA training.
- Nothing is hard-coded for scenario R. R, H and our own held-out scenarios are not used to choose settings (the ramp scale was chosen
  on 30 random scenarios).
- Classical refinement appears only as a separately labelled "hybrid" result, never in the raw QAOA counts.
- Files in `official/` are unmodified. Scenario H had not been released when we submitted; none of our results are on H.
- Known weaknesses (training hits the 300-evaluation cap, surrogate regret up to 32.6 % on some scenarios, selection noise in
  the ramp choice) are stated in `REPORT.md` section 7.
