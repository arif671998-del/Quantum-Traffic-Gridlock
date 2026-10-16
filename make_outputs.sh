#!/usr/bin/env bash
# Regenerate all submission outputs with one command:   bash make_outputs.sh
# - outputs/R/seed0..4   from official/scenario_R.json
# - outputs/H/seed0..4   from official/scenario_H.json (only if that file exists)
# - logs/versions.json   Python / library versions used
set -euo pipefail
cd "$(dirname "$0")"
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python

mkdir -p logs
$PY - <<'PYEOF'
import json, platform, numpy, scipy, qiskit
json.dump({"python": platform.python_version(), "platform": platform.platform(), "qiskit": qiskit.__version__,
           "numpy": numpy.__version__, "scipy": scipy.__version__}, open("logs/versions.json", "w"), indent=2)
PYEOF

for name in R H; do
  scenario="official/scenario_${name}.json"
  if [ -f "$scenario" ]; then
    for seed in 0 1 2 3 4; do
      $PY run.py --scenario "$scenario" --seed "$seed" --out "outputs/${name}/seed${seed}"
    done
  else
    echo "skipping $name: $scenario not found"
  fi
done
