# Prototype (numpy only, before implementation)
- `proto.py`: QUBO surrogate regret/correlation and QAOA (p = 1..4) lift/enrichment with exact probabilities,
  on scenario R + random scenarios from the section-7 ranges. Run: `python research/proto.py 20` (about 1 min).
- `proto2.py`: compares QAOA training variants (linear-ramp mean energy, multi-start, CVaR) at p = 4.
Results quoted in CLAUDE.md. Written with AI assistance (see AI_USE_LOG.md).
