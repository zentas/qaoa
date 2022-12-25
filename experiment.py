import numpy as np

from scipy.optimize import minimize

from qiskit import QuantumCircuit

import tqdm

import utils
import generate_random_matrix as grm


def compute_expectation(counts, model):
    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():
        obj = objective_function(bitstring, model)
        avg += obj * count
        sum_count += count

    return avg / sum_count


def objective_function(x, model):
    x = [int(bit) for bit in x]
    return model.decode_sample(x, vartype="BINARY").energy


class Experiment:
    def __init__(self, p, dim, method, create_backend, fixed_seed=None):
        self.p = p
        self.dim = dim
        self.create_backend = create_backend
        self.fixed_seed = fixed_seed
        self.method = method
        self.shots = 512

    def brute_force(self, model):
        def number2array(x):
            return [x // 2 ** (i - 1) % 2 for i in range(self.dim, 0, -1)]

        costs = [
            (
                "".join([str(n) for n in number2array(x)]),
                objective_function(number2array(x), model),
            )
            for x in range(2**self.dim)
        ]
        return dict(costs)

    def create_qaoa_circ(self, model, theta):
        nqubits = self.dim
        c = model.to_ising(index_label=True)
        p = len(theta) // 2  # number of alternating unitaries
        qc = QuantumCircuit(nqubits)

        beta = theta[:p]
        gamma = theta[p:]

        # initial_state
        for i in range(0, nqubits):
            qc.h(i)

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

        return qc

    def mesure_circ(self, qc):
        qc.measure_all()
        return qc

    def get_expectation_function(self, model):
        backend = self.create_backend()
        backend.shots = self.shots

        def execute_circ(theta):
            qc = self.mesure_circ(self.create_qaoa_circ(model, theta))
            counts = (
                backend.run(qc, seed_simulator=self.fixed_seed, nshots=self.shots)
                .result()
                .get_counts()
            )

            return compute_expectation(utils.invert_counts(counts), model)

        return execute_circ

    def generate_model(self, dens):
        return grm.model_from_sparse_matrix(grm.random_normal_matrix(self.dim, dens))

    def generate_close_matricies(self, N, center_point, radius, dens):
        return [center_point + radius * grm.random_normal_matrix(self.dim, dens).A for _ in range(N)]

    def generate_and_solve(self, dens):
        model = self.generate_model(dens)
        return self.solve(model)

    def solve(self, model, callback=None):
        method, options = self.method

        f = self.get_expectation_function(model)
        init = np.ones(shape=(2 * self.p,))
        return minimize(f, init, method=method, options=options, callback=callback)

    def validate(self, item, brute_solution):
        minval = min(brute_solution.values())
        res = [k for k, v in brute_solution.items() if v == minval]
        return item in res

    def solve_and_record_pathing(self, model):
        route, function_route = [], []

        expectation_function = self.get_expectation_function(model)

        def record_path_callback(x):
            route.append(x)
            function_route.append(expectation_function(x))

        res = self.solve(model, callback=record_path_callback)

        return res.x, route, function_route, res.fun

    def retrieve_state_distribution(self, model, theta):
        backend = self.create_backend()
        backend.shots = self.shots

        qc = self.mesure_circ(self.create_qaoa_circ(model, theta))
        res = backend.run(qc, seed_simulator=self.fixed_seed).result().get_counts()
        counts = utils.invert_counts(res)
        item, score = sorted(counts.items(), key=lambda x: x[1])[-1]

        return counts, (item, score)

    def compute_probability_distribution(
        self, models_and_brute_solutions, probability_retrieval
    ):
        res = []
        for model, brute_solution in tqdm.tqdm(models_and_brute_solutions):
            solution = self.solve(model)
            final_theta = solution.x
            counts, (item, _) = self.retrieve_state_distribution(model, final_theta)
            res.append(probability_retrieval(counts, brute_solution, item))
        return [prob for prob in res if prob is not None]

    def compute_probability_avg(
        self, models_and_brute_solutions, probability_retrieval
    ):
        res = self.compute_probability_distribution(
            models_and_brute_solutions, probability_retrieval
        )
        return sum(res) / len(res)

    def validate_fixed_prop(self, prob, dens):
        n = 0
        solved = 0
        while True:
            model = self.generate_model(dens=dens)

            solution = self.solve(model)
            final_theta = solution.x
            _, (item, _) = self.retrieve_state_distribution(model, final_theta)

            brute_solution = self.brute_force(model)
            if self.validate(item, brute_solution):
                solved += 1
            n += 1
            if solved / n > prob:
                break

        return n


# --- end of class Experiment


def retrieve_probabilities1(counts, brute_results, *args):
    minval = min(brute_results.values())
    res = [k for k, v in brute_results.items() if v == minval]
    if len(res) == 1:
        return counts.get(res[0], 0) / sum(counts.values())
    else:
        return None


def retrieve_probabilities2(counts, brute_results, *args):
    item, *_ = args
    minval = min(brute_results.values())
    res = [k for k, v in brute_results.items() if v == minval]
    if len(res) == 1 and item in res:
        return counts.get(res[0], 0) / sum(counts.values())
    else:
        return None


def retrieve_probabilities3(counts, brute_results, *args):
    minval = min(brute_results.values())
    res = [k for k, v in brute_results.items() if v == minval]
    c = 0
    for r in res:
        c += counts.get(r, 0)
    return c / sum(counts.values())
