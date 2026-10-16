# AI-assisted work log (Quantum Traffic Gridlock)

Rules (section 10): AI assistants are permitted; assistance must be declared and results independently verified.

| # | Date | Tool | What the AI did | How it was verified |
|---|------|------|-----------------|---------------------|
| 1 | 2026-10-16 | Claude (Claude app) | Explained the problem statement; checked the scorer reference vectors; wrote a numpy prototype (`research/`) comparing QUBO surrogates and QAOA training variants; wrote `traffic_qubo.py` (traffic recursion + local quadratic projection QUBO) and this project brief (`CLAUDE.md`). | Recursion equals official `scorer.py` on 7,936 schedule evaluations (R + 30 random scenarios); causal-window and projection-equality checks; R surrogate regret 0 %. |
| 2 | 2026-10-16 | Claude Code (Sonnet 5.5) | Wrote `tests/`, `qaoa.py`, `baselines.py`, `run.py`, `scenario_gen.py`; ran seeds 0-4 on R. | 19 unit tests (scorer reference vectors, recursion == scorer, QUBO == global quadratic fit, bit mapping, counts sum, circuit-vs-counts TV, determinism); `scorer.py` run on written `schedule.json`; metrics recomputed from `counts.json` in a test. |
