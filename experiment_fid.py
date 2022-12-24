
import numpy as np

from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity

import experiment_statvec as expr

class ExperimentStatvecFid(expr.ExperimentStatvec):
    def __init__(
        self,
        p,
        dim,
        method,
        create_backend=lambda: Aer.get_backend("statevector_simulator"),
        fixed_seed=None,
    ):
        super().__init__(p, dim, method, create_backend, fixed_seed)

    def create_qaoa_circ_fid(self, model, theta):
        nqubits = self.dim
        c = model.to_ising(index_label=True)
        p = len(theta) // 2  # number of alternating unitaries
        qc = QuantumCircuit(nqubits)
        sts = []

        beta = theta[:p]
        gamma = theta[p:]

        # initial_state
        for i in range(0, nqubits):
            qc.h(i)

        sts.append(Statevector.from_instruction(qc))

        for irep in range(0, p):
            # problem unitary: gamma
            # note: C is used here
            for (i, j) in c[1]:
                qc.rzz(2 * c[1][i, j] * gamma[irep], i, j)
            for i in c[0]:
                qc.rz(-2 * c[0][i] * gamma[irep], i)

            # mixer unitary: beta
            for i in range(0, nqubits):
                qc.rx(2 * beta[irep], i)

            sts.append(Statevector.from_instruction(qc))

        return sts

    def record_sts(self, model, theta):
        backend = self.create_backend()
        backend.shots = self.shots

        return self.create_qaoa_circ_fid(model, theta)

    def fidelity(self, sts):
        n = len(sts)
        fids = np.zeros((n, n))
        import itertools
        for (i, j) in itertools.product(range(n), range(n)):
            fids[i, j] = state_fidelity(sts[i], sts[j])
        return fids

    def solve_and_compute_fid(self, model):
        res = self.solve(model)
        return self.fidelity(self.record_sts(model, res.x))
