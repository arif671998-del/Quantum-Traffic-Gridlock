"""Tests for the QAOA pipeline (run.py). Runs the pipeline once on scenario R (seed 0, plus seed 0 again
and seed 1 for the determinism checks), then inspects the written files."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "official"))

import scorer
import qaoa
import run
from traffic_qubo import qubo_energy

SCEN = ROOT / "official" / "scenario_R.json"
R = json.loads(SCEN.read_text())


def load(d, name):
    return json.loads((Path(d) / name).read_text())


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.d0, cls.d0b, cls.d1 = base / "s0", base / "s0b", base / "s1"
        cls.m0 = run.run_pipeline(str(SCEN), 0, str(cls.d0))
        run.run_pipeline(str(SCEN), 0, str(cls.d0b))
        run.run_pipeline(str(SCEN), 1, str(cls.d1))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- required files ------------------------------------------------------------------
    def test_files_exist(self):
        for f in ["qubo.json", "circuit.qasm", "counts.json", "schedule.json", "metrics.json"]:
            self.assertTrue((self.d0 / f).exists(), f)

    def test_counts_sum_and_keys(self):
        counts = load(self.d0, "counts.json")
        self.assertEqual(sum(counts.values()), 4096)
        self.assertTrue(all(len(k) == 8 and set(k) <= {"0", "1"} for k in counts))
        self.assertTrue(all(isinstance(v, int) and v > 0 for v in counts.values()))

    def test_qasm3(self):
        text = (self.d0 / "circuit.qasm").read_text()
        self.assertTrue(text.startswith("OPENQASM 3"))
        self.assertIn("measure", text)
        self.assertNotIn("Parameter", text)          # all angles are bound numbers

    # ---- bit mapping ---------------------------------------------------------------------
    def test_bit_mapping_round_trip(self):
        for s in ["10101010", "11110000", "00000001", "10000000"]:
            self.assertEqual(run.schedule_to_bitstring(run.bitstring_to_schedule(s)), s)
        # first character of the bitstring = A slot 1 = schedule bit 0 = qubit 7 = clbit 7
        self.assertEqual(run.bitstring_to_schedule("10000000"), [1, 0, 0, 0, 0, 0, 0, 0])
        q = load(self.d0, "qubo.json")
        for v in q["variables"]:
            self.assertEqual(v["qubit"], 7 - v["schedule_bit"])
            self.assertEqual(v["clbit"], 7 - v["schedule_bit"])
        self.assertEqual(q["auxiliary_variables"], [])
        sch = load(self.d0, "schedule.json")
        self.assertIn("A slot 1", sch["bit_order"])

    def test_qubit_order_in_real_circuit(self):
        """Flip only schedule bit 0 (A slot 1) with an X on qubit 7: the sampled string must be 10000000."""
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler
        qc = qaoa.new_circuit()
        qc.x(7)
        qaoa.add_measurements(qc)
        res = StatevectorSampler(seed=1).run([qc], shots=16).result()[0].data.c.get_counts()
        self.assertEqual(res, {"10000000": 16})

    # ---- QUBO file -----------------------------------------------------------------------
    def test_qubo_json_energies(self):
        q = load(self.d0, "qubo.json")
        Q, off = np.array(q["Q"]), q["offset"]
        h, J, c = np.array(q["ising"]["h"]), np.array(q["ising"]["J"]), q["ising"]["offset"]
        for a in range(256):
            x = [(a >> (7 - j)) & 1 for j in range(8)]
            z = 1 - 2 * np.array(x)
            self.assertAlmostEqual(qubo_energy(Q, off, x), c + h @ z + z @ np.triu(J, 1) @ z, places=8)

    def test_qubo_is_best_quadratic_fit_of_official_cost(self):
        q = load(self.d0, "qubo.json")
        Q, off = np.array(q["Q"]), q["offset"]
        X = np.array([[(a >> (7 - j)) & 1 for j in range(8)] for a in range(256)], float)
        cols = [np.ones(256)] + [X[:, i] for i in range(8)] + [X[:, i] * X[:, j] for i in range(8) for j in range(i + 1, 8)]
        y = np.array([scorer.evaluate(R, [int(v) for v in x])["cost"] for x in X])
        coef, *_ = np.linalg.lstsq(np.stack(cols, 1), y, rcond=None)
        e = np.array([qubo_energy(Q, off, x) for x in X])
        self.assertLess(np.abs(e - np.stack(cols, 1) @ coef).max(), 1e-8)

    # ---- circuit / counts consistency ------------------------------------------------------
    def test_counts_match_circuit_statevector(self):
        """Rebuild the final circuit from the saved angles, compute exact probabilities and compare to counts.
        Threshold = 1.5 x the typical total-variation distance of an ideal 4096-shot sample (pure shot noise)."""
        m = load(self.d0, "metrics.json")
        q = load(self.d0, "qubo.json")
        qc = qaoa.bound_circuit(np.array(q["ising"]["h"]), np.array(q["ising"]["J"]), q["ising"]["sigma"],
                                m["qaoa"]["gammas"], m["qaoa"]["betas"])
        p = np.abs(Statevector(qc).data) ** 2           # index i <-> bitstring format(i, '08b')
        counts = load(self.d0, "counts.json")
        emp = np.array([counts.get(format(i, "08b"), 0) for i in range(256)]) / 4096
        tv = 0.5 * np.abs(emp - p).sum()
        rng = np.random.default_rng(0)
        ideal = [0.5 * np.abs(rng.multinomial(4096, p / p.sum()) / 4096 - p).sum() for _ in range(30)]
        self.assertLess(tv, 1.5 * np.mean(ideal), (tv, np.mean(ideal)))

    def test_qasm_text_has_the_same_angles(self):
        """circuit.qasm must be exactly the QASM 3 dump of the circuit rebuilt from qubo.json + saved angles
        (plus the final measurements)."""
        m = load(self.d0, "metrics.json")
        q = load(self.d0, "qubo.json")
        qc = qaoa.bound_circuit(np.array(q["ising"]["h"]), np.array(q["ising"]["J"]), q["ising"]["sigma"],
                                m["qaoa"]["gammas"], m["qaoa"]["betas"])
        qaoa.add_measurements(qc)
        from qiskit import qasm3
        self.assertEqual(qasm3.dumps(qc), (self.d0 / "circuit.qasm").read_text())

    # ---- budget, determinism, limits -------------------------------------------------------
    def test_training_budget(self):
        m = load(self.d0, "metrics.json")
        self.assertLessEqual(m["qaoa"]["training_evaluations"], 300)
        self.assertEqual(m["qaoa"]["p"], 4)
        self.assertLessEqual(m["qaoa"]["n_qubits"], 16)

    def test_same_seed_same_counts(self):
        self.assertEqual((self.d0 / "counts.json").read_text(), (self.d0b / "counts.json").read_text())
        self.assertEqual((self.d0 / "circuit.qasm").read_text(), (self.d0b / "circuit.qasm").read_text())

    def test_different_seed_different_counts(self):
        self.assertNotEqual(load(self.d0, "counts.json"), load(self.d1, "counts.json"))

    def test_metrics_consistent_with_counts(self):
        """E_Q, lift and enrichment in metrics.json must be reproducible from counts.json + scorer.py."""
        counts = load(self.d0, "counts.json")
        costs = {s: scorer.evaluate(R, s)["cost"] for s in (format(i, "08b") for i in range(256))}
        eq = sum(costs[s] * n for s, n in counts.items()) / 4096
        cstar, crand = min(costs.values()), np.mean(list(costs.values()))
        m = load(self.d0, "metrics.json")
        self.assertAlmostEqual(m["E_Q"], eq, places=9)
        self.assertAlmostEqual(m["lift"], (crand - eq) / (crand - cstar), places=9)
        elite = sorted(costs, key=lambda s: (costs[s], s))[:13]
        self.assertAlmostEqual(m["elite_enrichment"], sum(counts.get(s, 0) for s in elite) / 4096 / (13 / 256), places=9)


if __name__ == "__main__":
    unittest.main()
