import numpy as np

from scipy import integrate, optimize

import distributions as dist


# --------------------------------------------------------------------------------
# functions
# --------------------------------------------------------------------------------
def bin_values(f, args, bin_edges):
    values = np.zeros(len(bin_edges)-1)
    for i in range(len(bin_edges)-1):
        a, b = bin_edges[i], bin_edges[i+1]
        values[i] = integrate.quad(lambda x: f(x, *args), a, b)[0]

    return values


def fit(f, bin_edges, prob, p0):
    # each bin is the avg of the edges.
    bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def g(x, *args):
        n = np.digitize(x, bin_edges)
        values = bin_values(f, args, bin_edges)[n-1]
        return values
    
    fitvals = optimize.curve_fit(g, bins, prob, p0)

    return fitvals
