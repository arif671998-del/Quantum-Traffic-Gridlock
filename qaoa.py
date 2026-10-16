"""QAOA for the 8-bit traffic QUBO (Qiskit 2.x).

Qubit mapping (important, tested): schedule bit j (0 = "A slot 1" ... 7 = "B slot 4") lives on qubit 7 - j and is
measured into classical bit 7 - j.  Qiskit prints classical bit 0 on the RIGHT, so the printed bitstring reads
directly as the schedule string "A1 A2 A3 A4 B1 B2 B3 B4".

Cost Hamiltonian: C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j  (from to_ising; Z=+1 means bit 0).
We divide by sigma, the standard deviation of C over uniformly random bitstrings, so the angles gamma mean
the same thing for every scenario (sigma^2 = sum h_i^2 + sum J_ij^2 exactly for an Ising form).
"""
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

N = 8          # qubits = decision bits (no auxiliary variables)
P = 4          # QAOA depth (limit is 4)
MAX_EVALS = 300


def qubit_of(j):
    """Schedule bit index -> qubit index."""
    return N - 1 - j


def ising_sigma(h, J):
    """Std-dev of the Ising energy over uniform random bitstrings (analytic)."""
    return float(np.sqrt(np.sum(np.asarray(h) ** 2) + np.sum(np.triu(np.asarray(J), 1) ** 2)))


def new_circuit():
    """Empty 8-qubit circuit with an 8-bit classical register called 'c'."""
    return QuantumCircuit(QuantumRegister(N, "q"), ClassicalRegister(N, "c"))


def qaoa_circuit(h, J, sigma, p=P):
    """Parametrised QAOA circuit (no measurements). Returns (circuit, gammas, betas) with ParameterVectors.
    Layer k: RZ(2 gamma_k h_i/sigma) on every qubit, RZZ(2 gamma_k J_ij/sigma) on every coupled pair, RX(2 beta_k).
    """
    gam, bet = ParameterVector("gamma", p), ParameterVector("beta", p)
    qc = new_circuit()
    qc.h(range(N))                                           # uniform superposition over all 256 schedules
    for k in range(p):
        for j in range(N):
            if h[j] != 0:
                qc.rz(2 * gam[k] * float(h[j]) / sigma, qubit_of(j))
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0:
                    qc.rzz(2 * gam[k] * float(J[i, j]) / sigma, qubit_of(i), qubit_of(j))
        qc.rx(2 * bet[k], range(N))
    return qc, gam, bet


def bound_circuit(h, J, sigma, gammas, betas):
    """The same circuit with numeric angles (no measurements)."""
    qc, gam, bet = qaoa_circuit(h, J, sigma, len(gammas))
    return qc.assign_parameters({**dict(zip(gam, gammas)), **dict(zip(bet, betas))})


def add_measurements(qc):
    """Measure qubit q into clbit q (so qubit 7 - j -> clbit 7 - j for schedule bit j)."""
    qc.measure(range(N), range(N))
    return qc


def energy_table(h, J, sigma):
    """Normalised Ising energy for each of the 256 basis states; index i <-> bitstring format(i,'08b')."""
    e = np.zeros(2 ** N)
    for i in range(2 ** N):
        z = np.array([1 - 2 * ((i >> (N - 1 - j)) & 1) for j in range(N)])    # z_j for schedule bit j
        e[i] = (h @ z + z @ np.triu(J, 1) @ z) / sigma
    return e


def initial_angles(p=P, gamma_scale=0.6, beta_scale=0.6):
    """Linear ramp: gamma_k = gamma_scale (k-0.5)/p rising, beta_k = beta_scale (1-(k-0.5)/p) falling, k=1..p.
    Default scales 0.6 / 0.6 (the scales are varied only in the ablation study)."""
    k = np.arange(1, p + 1)
    return gamma_scale * (k - 0.5) / p, beta_scale * (1 - (k - 0.5) / p)


def train(h, J, sigma, p=P, init=None, maxiter=MAX_EVALS, rhobeg=0.2, cvar_alpha=None):
    """COBYLA on the exact (statevector) expectation of the normalised cost. No shots, no randomness.
    cvar_alpha=None (default) minimises the mean energy; a value such as 0.25 minimises the CVaR instead
    (mean energy of the lowest-energy 25 % of the probability mass) - used only in the ablation.
    Returns dict with gammas, betas, objective history and the number of objective calls."""
    qc, gam, bet = qaoa_circuit(h, J, sigma, p)
    diag = energy_table(h, J, sigma)
    history = []
    order = np.argsort(diag, kind="stable")

    def objective(theta):
        history.append(None)                                   # count every call
        assert len(history) <= MAX_EVALS, "training budget exceeded"
        bound = qc.assign_parameters({**dict(zip(gam, theta[:p])), **dict(zip(bet, theta[p:]))})
        probs = np.abs(Statevector(bound).data) ** 2
        if cvar_alpha is None:
            val = float(probs @ diag)
        else:
            ps = probs[order]
            w = np.clip(cvar_alpha - (np.cumsum(ps) - ps), 0, ps)     # probability mass taken from each sorted state
            val = float(w @ diag[order] / cvar_alpha)
        history[-1] = val
        return val

    g0, b0 = initial_angles(p) if init is None else init
    x0 = np.concatenate([g0, b0])
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": rhobeg})
    assert len(history) <= MAX_EVALS
    return {"gammas": res.x[:p].tolist(), "betas": res.x[p:].tolist(), "history": history,
            "evaluations": len(history), "initial_objective": history[0], "final_objective": float(res.fun)}


def sample(circuit, seed, shots=4096):
    """Measure the bound circuit with a seeded sampler. Returns {bitstring: count} (8-char keys)."""
    qc = add_measurements(circuit.copy())
    counts = StatevectorSampler(seed=seed).run([qc], shots=shots).result()[0].data.c.get_counts()
    return {k: int(v) for k, v in sorted(counts.items())}, qc
