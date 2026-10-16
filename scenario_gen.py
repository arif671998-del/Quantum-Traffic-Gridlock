"""Random scenarios drawn from the hidden-scenario ranges of problem statement section 7.

Used only by the tests and the generalisation study - never by run.py.
"""
import numpy as np


def random_scenario(rng, name="rand"):
    """arrivals 0-8, capacity 4-6, storage 5-8, overflow weight 2-6, switch weight 0.25-1.5,
    one A->B link on NS or EW with fraction 0.4-0.8."""
    arr = {n: {ph: [int(v) for v in rng.integers(0, 9, 4)] for ph in ("NS", "EW")} for n in ("A", "B")}
    return {
        "name": name,
        "arrivals": arr,
        "service_capacity": int(rng.integers(4, 7)),
        "storage_capacity": int(rng.integers(5, 9)),
        "overflow_weight": int(rng.integers(2, 7)),
        "switch_weight": float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])),
        "link": {"from": "A", "to": "B", "phase": str(rng.choice(["NS", "EW"])),
                 "fraction": round(float(rng.uniform(0.4, 0.8)), 2)},
    }
