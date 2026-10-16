# Quantum Traffic Gridlock — QUBO + QAOA for two coordinated intersections

IBM Qiskit Fall Fest 2026 · QC@IITI, IIT Indore · problem statement 2. Team: Arif, Rohit, Mehul.

This project takes a traffic scenario (two intersections A and B, four 30-second slots, one A→B link), builds a QUBO
from the traffic queue recursion with a documented, scenario-independent rule, converts it to an Ising Hamiltonian,
runs a depth-4 QAOA on 8 qubits in Qiskit (noiseless statevector simulation, CPU only), and writes the sampled schedule
distribution (4,096 shots). The QUBO is never built from the optimal schedule; enumerating all 256 schedules is used only
for benchmarks (C\*, C_rand, elite set). The full derivation, benchmarks and limitations are in [`REPORT.md`](REPORT.md);
AI assistance is declared in [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md).

> **Scenario H:** the organisers released scenario H late (about 17:40 IST on 16 Oct 2026, Discord #doubts). The
> pipeline ran on it unchanged: `python run.py --scenario official/scenario_H.json --seed N --out outputs/H/seedN`
> (`make_outputs.sh` does this automatically). Results for H are in the table below and in `REPORT.md` section 4.

## Install

Python 3.10+ (tested on 3.13.9; exact library versions in `requirements.txt` and `logs/versions.json`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # qiskit 2.x, numpy, scipy — nothing else is needed
```

## Run

```bash
python run.py --scenario official/scenario_R.json --seed 0 --out outputs/R/seed0
```

No manual changes are needed for another scenario file (same JSON format as `official/scenario_R.json`).
The training does not depend on `--seed`; the seed only sets the final 4,096-shot sampler, so a given seed always
reproduces the same files.

### Output files

| file | content |
|---|---|
| `qubo.json` | QUBO matrix `Q` (upper triangle + diagonal) and `offset`; Ising `h`, `J`, `offset`, and the normaliser `sigma`; the variable → schedule-bit → qubit → classical-bit mapping; `"auxiliary_variables": []` (no auxiliary variables, 8 qubits). |
| `circuit.qasm` | The final bound-parameter QAOA circuit including measurements, OpenQASM 3 (`qiskit.qasm3.dumps`). |
| `counts.json` | Exactly 4,096 raw shots of that circuit as `{bitstring: count}` (nothing else in the file). |
| `schedule.json` | Decoded schedules with the bit order spelled out: `primary` (raw QAOA sample with the lowest QUBO energy), `most_frequent` (raw QAOA), and `hybrid` (greedy 1-bit-flip refinement of the primary on the QUBO, **labelled as not raw QAOA**), each with its official cost. A top-level `"bits"` field (= primary) lets `official/scorer.py` read the file directly. |
| `metrics.json` | Our diagnostics: C\*, C_rand, E_Q, lift, elite enrichment, the section-9 baselines, QUBO regret, training evaluations, trained angles, runtime, library versions. |

**Bit order and qubit mapping.** A schedule is 8 bits: A slot 1, A slot 2, A slot 3, A slot 4, B slot 1, …, B slot 4
(left to right); 1 = North–South green, 0 = East–West green. Schedule bit *j* (0 = A slot 1 … 7 = B slot 4) is carried by
qubit 7 − *j* and measured into classical bit 7 − *j*. Qiskit prints classical bit 0 on the right, so every key in
`counts.json` reads directly as the schedule string, e.g. `"11101010"` = A `1110`, B `1010`. This is tested.

### Check with the official scorer

```bash
python official/scorer.py --scenario official/scenario_R.json --schedule outputs/R/seed0/schedule.json
```

(`official/` is the organisers' folder and is never modified.)

## Reproduce everything

| command | what it does |
|---|---|
| `bash make_outputs.sh` | Regenerates `outputs/R/seed0..4` and `outputs/H/seed0..4` (H only if `official/scenario_H.json` exists) and `logs/versions.json`. |
| `python -m unittest discover -s tests` | 20 tests (about 4 s): scorer reference vectors, our recursion = scorer, QUBO = best quadratic fit, bit-mapping round trip, counts sum 4,096, circuit vs counts, ≤ 300 evaluations, initial angles better than random, same seed → identical counts. |
| `python ablations.py` | Depth / initialisation / surrogate degree / objective on R and 30 random scenarios → `logs/ablations.{json,md}` (about 3.5 min). |
| `python generalisation.py` | 100 random scenarios through the same pipeline function → `logs/generalisation.{json,md}`. |
| `python report_numbers.py` | Re-checks recursion = scorer on 7,936 evaluations and collects the numbers quoted in the report → `logs/verification.json`, `logs/report_numbers.json`. |

## Limits

| item | used | limit |
|---|---|---|
| qubits | 8 | 16 |
| QAOA depth p | 4 | 4 |
| training objective evaluations | 300 | 300 |
| final shots | 4,096 | exactly 4,096 |
| runtime per seed | about 1.3 s (1.28–1.39 s on R, 1.27–1.42 s on H) | 10 min |
| determinism | same seed → byte-identical `counts.json` and `circuit.qasm` (tested) | required |
| resources | CPU only, no network, well under 8 GB | CPU, 8 GB |

## Results summary

All values are read from `outputs/*/seed*/metrics.json` (QAOA p = 4, ramp initialisation γ scale 1.0, β scale −0.4; see `REPORT.md` section 3 for why β is negative).

| scenario | C\* | C_rand | seed 0–4: E_Q range | seed 0–4: lift | min lift | enrichment range | QUBO regret |
|---|---|---|---|---|---|---|---|
| **R** (official public) | 72.0 | 151.25 | 89.21–89.42 | 0.7802–0.7829 | **0.7802** | 11.64–12.02 | 0 % |
| **H** (official shared, run unmodified) | 234.0 | 300.28 | 242.83–243.28 | 0.8600–0.8668 | **0.8600** | 16.30–16.39 | 0.86 % |
| heldout_1 (our own test) | 188.0 | 227.36 | 194.52–194.71 | 0.8296–0.8343 | 0.8296 | 12.99–13.05 | 0 % |
| heldout_2 (our own test) | 277.5 | 318.22 | 285.30–285.50 | 0.8035–0.8085 | 0.8035 | 5.74–6.00 | 0 % |

`heldout_1/2` are random scenarios we generated (`scenarios/`, `scenario_gen` seed 99); they are **not** the official H.

Per-seed values and the section-9 baselines (alternating, greedy, enumeration, random, QUBO minimiser, raw QAOA, labelled hybrid) are
in `REPORT.md` section 4. Official costs of the baselines on H: best alternating 261.0, greedy 244.0, QUBO minimiser 236.0, optimum 234.0.

Over 100 random scenarios (seed 0, `logs/generalisation.md`): lift mean 0.818, min 0.671, 10th percentile 0.748;
90 % of scenarios have QUBO regret ≤ 5 % (worst 32.6 %).

## File layout

| path | purpose |
|---|---|
| `run.py` | The required command-line pipeline (`run_pipeline` is also imported by the tests and `generalisation.py`). |
| `traffic_qubo.py` | Queue recursion, causal windows, local quadratic projection, QUBO and Ising conversion. |
| `qaoa.py` | Circuit construction, qubit mapping, COBYLA training on the exact expectation, seeded sampling. |
| `baselines.py` | Section-9 baselines, lift / enrichment, QUBO minimiser, hybrid refinement (all costs via `official/scorer.py`). |
| `scenario_gen.py` | Random scenarios from the section-7 ranges (tests and studies only). |
| `ablations.py`, `generalisation.py`, `report_numbers.py` | Experiments and number collection. |
| `make_outputs.sh`, `requirements.txt` | Reproducibility. |
| `tests/` | `test_qubo.py`, `test_pipeline.py`. |
| `official/` | Organisers' `scorer.py`, `scenario_R.json`, `scenario_H.json`, README (unmodified). |
| `scenarios/` | Our two held-out test scenarios. |
| `outputs/R/seed0..4`, `outputs/H/seed0..4`, `outputs/heldout_1/`, `outputs/heldout_2/` | Generated outputs. |
| `logs/` | `ablations`, `generalisation`, `verification`, `report_numbers`, `versions`. |
| `research/` | Early numpy prototype used to compare QUBO surrogates and training variants. |
| `CLAUDE.md`, `AI_USE_LOG.md`, `AI_DISCLOSURE.md` | Project brief and AI-use records. |
| `REPORT.md` | Technical report (≤ 3 pages). |
