# Quantum Traffic Gridlock (problem statement 2) — project brief for Claude Code

Team: Arif, Rohit, Mehul (beginners in quantum computing). Explain each step in plain words, say what output to
expect before running anything, and keep the code simple and commented (judges review it).

**Deadline:** Hour 24 = Fri 16 Oct 2026, about 21:30 IST (team will confirm). **Scenario H** (shared evaluation
scenario) is released at Hour 18, about 15:30 IST: the pipeline must run on it unmodified.

## Source of truth
- `PROBLEM_STATEMENT.txt` (full official text, sections 1–13). Re-read the relevant section before each decision.
- `official/scorer.py` is the authoritative evaluator. **Never modify anything in `official/`.**
- `official/scenario_R.json` is the public scenario. Reference checks: 10101010 → 116.0, 11111111 → 146.0,
  00000000 → 237.0, 01111001 → 92.5.

## What must exist (sections 6, 12)
`python run.py --scenario <scenario.json> --seed <int> --out <dir>` with no manual changes, writing:

| file | content |
|---|---|
| `qubo.json` | QUBO matrix + offset AND Ising h, J, offset; variable → schedule-bit → qubit mapping; `auxiliary_variables: []` |
| `circuit.qasm` | final circuit with bound parameters, OpenQASM 3 (`qiskit.qasm3.dumps`) |
| `counts.json` | exactly 4,096 raw final-circuit shots, `{bitstring: count}` only, sum must be 4096 |
| `schedule.json` | decoded schedule(s) with the bit order spelled out (A slots 1–4 then B slots 1–4; 1 = NS, 0 = EW) |
| `metrics.json` (ours) | C*, C_rand, E_Q, lift, elite enrichment, baselines, training evaluations used, runtime, versions |

Limits: ≤ 16 qubits, QAOA depth p ≤ 4, ≤ 300 training objective evaluations, exactly 4,096 final shots,
deterministic for a given seed, CPU only, no network, ≤ 10 min per seed, 8 GB. Classical refinement only as a
separately labelled "hybrid" result, never mixed into the raw QAOA counts. Must not hard-code R.

Also: public GitHub repo, outputs for R and H committed, report ≤ 3 pages, 1-page AI-use disclosure, tagged release.

## Scoring (section 13, 100 marks) — what to optimise
Formulation and QUBO 25 (derivation 15; surrogate regret 10: full at ≤ 5 %, zero at ≥ 25 %) · QAOA implementation 20
(reproducible 8; circuit/QUBO/counts consistency 6; design choices 6) · hidden-scenario sampling quality 25 (lift 12,
elite enrichment 8, worst-seed robustness 5) · generalisation 15 (unmodified on H and S 9; ablations over depth,
penalties, initialisation 6) · extensions 10 · report 5.

Lift = (C_rand − E_Q)/(C_rand − C*), E_Q = mean OFFICIAL cost of the 4,096 raw samples. Elite = the 13 lowest-cost
schedules (ties by ascending bitstring); enrichment = shot fraction in the elite set / (13/256). Judges take the
minimum lift over ≥ 5 seeds.

## Design (already chosen and checked by a numpy prototype — implement this)
**QUBO (file `traffic_qubo.py`, already written and tested — read its docstring; reuse it, don't rewrite it).**
8 binary variables, no auxiliary variables (8 qubits). The cost is a sum of local terms per (intersection,
approach, slot); each term depends only on its causal window of bits; each term is replaced by its best quadratic
approximation (degree-2 Walsh–Fourier truncation over its window); the switching cost is added exactly. Checks done:
- our recursion `official_cost` equals `scorer.py` on all 256 schedules of R and 30 random scenarios (7,936/7,936);
- changing bits outside a term's causal window never changes the term;
- the summed local projections equal the global best quadratic fit to 1e-12 (so the QUBO is the best L2 quadratic
  surrogate, but it is built only from the traffic recursion, never from the global optimum);
- R: QUBO minimiser 11101010 has official cost 72.0 = C* (regret 0 %), correlation(QUBO energy, official cost) 0.995.
  Ablation: linear-only surrogate (`build_qubo(sc, degree=1)`) picks 11110000 (cost 134, regret 86 %).
- 40 random scenarios in the hidden ranges: mean regret 0.7 %, 95 % within 5 %, mean correlation 0.994.

**QAOA (to implement in Qiskit 2.x).**
- Qubit mapping: schedule bit j ↔ qubit 7 − j, measured into clbit 7 − j, so Qiskit's bitstring (clbit 0
  rightmost) reads directly as the schedule string "A1 A2 A3 A4 B1 B2 B3 B4". Write this mapping into qubo.json and
  schedule.json and test it.
- Cost Hamiltonian from `to_ising(Q, offset)` (x = (1 − z)/2). Normalise energies by their standard deviation under
  the uniform distribution, which is analytic for an Ising form: sigma = sqrt(sum h_i² + sum_{i<j} J_ij²).
  Layer k: RZ(2·γ_k·h_i/σ) on each qubit, RZZ(2·γ_k·J_ij/σ) on each pair with J_ij ≠ 0, then RX(2·β_k) on each qubit.
  Start from H on all qubits.
- p = 4. Initial angles: linear ramp γ_k = 1.0·(k − 0.5)/p, β_k = 0.4·(1 − (k − 0.5)/p), k = 1..p (scales 1.0/0.4 chosen on 30 random scenarios by the pre-registered rule in `ablations.py`, 16 Oct; originally 0.6/0.6).
- Training objective: exact expectation of the normalised cost Hamiltonian from the statevector (no shots),
  COBYLA (scipy) with maxiter 300, rhobeg 0.2; count every objective call and assert ≤ 300.
  Training does not depend on the seed (so worst-seed lift is stable); the seed only sets the final sampler.
- Final: bind parameters, add measurements, sample exactly 4,096 shots with a seeded sampler
  (`qiskit.primitives.StatevectorSampler(seed=seed)`), write counts.json; export the same circuit to circuit.qasm.
- Prototype results (numpy, exact probabilities): R lift 0.72, enrichment 9.5×; over 40 random hidden-range
  scenarios lift mean 0.77 (min 0.59), enrichment mean 9.4×. p = 4 mean-energy beat CVaR and multi-start variants.
- schedule.json: primary = the sampled schedule with the lowest QUBO energy (label "raw QAOA"); also report the most
  frequent sample; optional "hybrid (classical refinement, NOT raw QAOA)" = greedy 1-bit-flip descent on the QUBO
  from that sample. Report each one's official cost.

## Baselines and diagnostics (section 9) — compute in run.py, save to metrics.json
(1) best of the 4 alternating schedules (A ∈ {1010, 0101} × B ∈ {1010, 0101}); (2) per-slot greedy: at each slot each
intersection serves the approach with the larger queue after arrivals, ties to NS, simulated forward with transfers;
(3) exact enumeration of 256 (C*); (4) uniform-random mean (C_rand); (5) exact minimiser of our QUBO, evaluated by the
official scorer (surrogate ceiling, regret %); (6) raw QAOA (E_Q, lift, enrichment, best sample) and the labelled
hybrid. Enumeration is allowed as a benchmark only — it must never feed into the QUBO or the QAOA training.

## Experiments for the report
- R and H, seeds 0–4 each: lift, enrichment, min lift, runtime. Commit outputs as `outputs/R/seed0/…`, `outputs/H/…`.
- Ablations: depth p = 1, 2, 3, 4; initialisation (linear ramp vs random angles); surrogate degree 1 vs 2
  ("penalty"/weighting variants if time allows).
- Extension (generalisation evidence): 100 random scenarios drawn from the section-7 ranges (fixed seed):
  regret and lift distributions. Keep a script for it.

## Tests (write first, keep fast)
reference vectors via scorer; recursion == scorer on all 256 (R + random); qubo.json energies == official
decomposition of the QUBO; bit mapping round trip (bitstring → schedule); counts sum 4096 with 8-bit keys;
circuit consistency (statevector of the bound circuit vs counts, e.g. total variation < 0.05);
≤ 300 evaluations; same seed → identical counts.

## Working rules
- Ask before installing anything beyond qiskit, numpy, scipy. Use `.venv` (qiskit 2.x).
- After every step: run tests, show numbers, update the status list below.
- Every report number must come from files in `outputs/` or `logs/` (generated by script, not typed).
- Log AI-assisted work in `AI_USE_LOG.md` (required disclosure).

## Status
- [x] Traffic recursion + QUBO builder (`traffic_qubo.py`) verified in a numpy prototype (see `research/`)
- [x] Tests (`.venv/bin/python -m unittest discover -s tests`, 19 tests, ~4 s; unittest because pytest is not installed)
- [x] QAOA + run.py with the four required outputs (`qaoa.py`, `run.py`)
- [x] Baselines + metrics.json (`baselines.py`)
- [x] Runs on R, seeds 0-4 (`outputs/R/seed0..4`); [ ] H when released (~15:30): `python run.py --scenario <H.json> --seed S --out outputs/H/seedS`
- [x] Ablations (`ablations.py` -> `logs/ablations.md`; ramp scale changed to 1.0/0.4 by the pre-set rule)
- [x] Generalisation study (`generalisation.py` -> `logs/generalisation.md`)
- [ ] README, report ≤ 3 pages, AI disclosure, tag
