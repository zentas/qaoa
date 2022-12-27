from scipy.stats import rv_continuous
from scipy.sparse import random
from numpy.random import default_rng

from pyqubo import Binary
from pyqubo import Array, Constraint


class CustomDistribution(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)


def random_normal_matrix(dim, dens):
    X = CustomDistribution()
    Y = X()
    return random(dim, dim, dens, random_state=default_rng(), data_rvs=Y.rvs)


def random_even_matrix(dim, dens):
    return random(dim, dim, dens)


def model_from_sparse_matrix(sparse):
    dim = sparse.get_shape()[0]
    x = Array.create("x", shape=(dim), vartype="BINARY")
    constr = 0
    dok = sparse.todok()
    for (i, j) in dok.keys():
        constr += x[int(i)] * x[int(j)] * dok[(i, j)]
    model = Constraint(constr, "").compile()
    return model


def model_from_matrix(matrix):
    dim = matrix.shape[0]
    assert matrix.shape[0] == matrix.shape[1], "matrix must be quadric"
    x = Array.create("x", shape=(dim), vartype="BINARY")
    constr = 0
    for i in range(dim):
        for j in range(i, dim):
            constr += x[i] * x[j] * matrix[i, j]
    model = Constraint(constr, "").compile()
    return model
