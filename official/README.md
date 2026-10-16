# Quantum Traffic Gridlock — official evaluation files

This folder contains only the **traffic evaluator** and the **public scenario R**. It does not provide a QUBO, QAOA implementation, optimizer, or optimal schedule.

## Files
- `scorer.py` — authoritative schedule evaluator, Python 3.9+ standard library only.
- `scenario_R.json` — public scenario matching the participant problem statement.

## Run

```bash
python scorer.py --scenario scenario_R.json --schedule 10101010
python scorer.py --scenario scenario_R.json --schedule 01111001
```

Bit order: four A slots followed by four B slots. `1` means NS green, `0` means EW green. `--schedule` may also be a JSON path containing `{"bits":"10101010"}`, an 8-integer array, or `{"A":[1,0,1,0],"B":[1,0,1,0]}`.

## Reference checks
- `10101010` → 116.0
- `11111111` → 146.0
- `00000000` → 237.0
- `01111001` → 92.5

The scorer models integer vehicle counts and rounds transfers with decimal half-up (`floor(fraction * departures + 0.5)`). All eight-bit schedules are valid. It returns cost, delay, overflow, switches, and final queues. Teams must independently derive the optimization model and quantum implementation.
