import numpy as np

import matplotlib.pyplot as plt

def compute_energy_landscape(expectation_function, grid):
    N, M = grid

    beta = [2 * np.pi * i / N - np.pi for i in range(0, N)]
    gamma = [2 * np.pi * i / M - np.pi for i in range(0, M)]

    beta, gamma = np.meshgrid(beta, gamma)
    F = np.array([
        [expectation_function([b_entry] + [g_entry]) for (b_entry, g_entry) in zip(b, g)]
        for (b, g) in zip(beta, gamma)
    ])

    return beta, gamma, F


def plot_energy_landscape(landscape, pathing=None):
    beta, gamma, F = landscape

    fig, ax = plt.subplots()
    c = ax.pcolormesh(beta, gamma, F)
    fig.colorbar(c, ax=ax)

    if pathing is not None:
        final_theta, route_theta, route_function, *_ = pathing

        [ax.plot(x, y, 'mo') for (x, y) in route_theta]
        ax.plot(final_theta[0], final_theta[1], 'go')

        [ax.annotate("{:2.2f}".format(f), (x, y))
         for ((x, y), f) in zip(route_theta, route_function)]

    fig.set_size_inches(6, 5, forward=True)
    plt.xlabel('beta')
    plt.ylabel('gamma')
    plt.show()


def plot_energy_route(route_function, final_function):
    fig, ax = plt.subplots()
    ax.plot(route_function + [final_function])
    print(final_function)
    fig.set_size_inches(6, 5, forward=True)
    plt.show()


def analazing_the_result(counts):
    plt.bar(counts.keys(), counts.values())
    print(counts)


def hist_with_avg(x):
    plt.hist(x, bins=30)
    xx = np.array(x).mean()
    plt.axvline(xx, color='k', linestyle='dashed', linewidth=2)
    min_ylim, max_ylim = plt.ylim()
    plt.text(xx*1.1, max_ylim*0.9, 'Среднее: {:.3f}'.format(xx))
    plt.show()