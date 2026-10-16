"""Tests for the traffic model and the QUBO (no quantum code involved).

Run all tests:  .venv/bin/python -m unittest discover -s tests -v
"""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official"))

import scorer                                   # official evaluator (never modified)
from scenario_gen import random_scenario
from traffic_qubo import build_qubo, official_cost, qubo_energy, to_ising

R = json.loads((ROOT / "official" / "scenario_R.json").read_text())
ALL = [list(b) for b in product([0, 1], repeat=8)]


def scorer_cost(sc, x):
    return scorer.evaluate(sc, x)["cost"]


class ReferenceVectors(unittest.TestCase):
    """The four numbers printed in the problem statement must come out of scorer.py."""

    def test_vectors(self):
        for bits, want in [("10101010", 116.0), ("11111111", 146.0), ("00000000", 237.0), ("01111001", 92.5)]:
            self.assertEqual(scorer.evaluate(R, bits)["cost"], want, bits)


class RecursionMatchesScorer(unittest.TestCase):
    """Our own queue recursion (used to build the QUBO) must equal scorer.py on all 256 schedules."""

    def test_R_and_random(self):
        rng = np.random.default_rng(123)
        for sc in [R] + [random_scenario(rng) for _ in range(10)]:
            for x in ALL:
                self.assertAlmostEqual(official_cost(sc, x), scorer_cost(sc, x), places=9)


class QuboIsBestQuadraticFit(unittest.TestCase):
    """The summed local projections must equal the global least-squares quadratic fit of the
    official cost (decomposition check), and the Ising form must give identical energies."""

    def check(self, sc):
        Q, off = build_qubo(sc)
        X = np.array(ALL, float)
        cols = [np.ones(256)] + [X[:, i] for i in range(8)] + [X[:, i] * X[:, j] for i in range(8) for j in range(i + 1, 8)]
        y = np.array([scorer_cost(sc, x) for x in ALL])
        coef, *_ = np.linalg.lstsq(np.stack(cols, 1), y, rcond=None)
        fit = np.stack(cols, 1) @ coef
        e = np.array([qubo_energy(Q, off, x) for x in ALL])
        self.assertLess(np.abs(e - fit).max(), 1e-8)
        h, J, c = to_ising(Q, off)
        for x, ex in zip(ALL, e):
            z = 1 - 2 * np.array(x)
            self.assertAlmostEqual(c + h @ z + z @ np.triu(J, 1) @ z, ex, places=9)

    def test_R(self):
        self.check(R)

    def test_random(self):
        rng = np.random.default_rng(7)
        for _ in range(3):
            self.check(random_scenario(rng))

    def test_R_minimiser_is_optimal(self):
        Q, off = build_qubo(R)
        best = min(ALL, key=lambda x: qubo_energy(Q, off, x))
        self.assertEqual("".join(map(str, best)), "11101010")
        self.assertEqual(scorer_cost(R, best), 72.0)


if __name__ == "__main__":
    unittest.main()
