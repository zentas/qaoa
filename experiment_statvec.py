import experiment
import utils
from qiskit import Aer, execute

# вместо статистики -- амплитуды
def compute_expectation_statevector(svd, model):
    avg = 0
    for bitstring, prob in svd.items():
        obj = experiment.objective_function(bitstring, model)
        avg += obj * prob
    return avg


class ExperimentStatvec(experiment.Experiment):
    def __init__(
        self,
        p,
        dim,
        method,
        create_backend=lambda: Aer.get_backend("statevector_simulator"),
        fixed_seed=None,
    ):
        super().__init__(p, dim, method, create_backend, fixed_seed=fixed_seed)

    def get_expectation_function(self, model):
        backend = self.create_backend()

        def execute_circ(theta):
            qc = self.create_qaoa_circ(model, theta)
            sv = execute(qc, backend).result().get_statevector()

            return compute_expectation_statevector(
                utils.invert_counts(sv.probabilities_dict()), model
            )

        return execute_circ

    def retrieve_state_distribution(self, model, theta):
        backend = self.create_backend()
        backend.shots = self.shots

        qc = self.create_qaoa_circ(model, theta)
        sv = execute(qc, backend).result().get_statevector()
        item, score = sorted(utils.invert_counts(sv.probabilities_dict()).items(), key=lambda x: x[1])[-1]

        return utils.invert_counts(sv.probabilities_dict()), (item, score)

