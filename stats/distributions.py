import numpy as np

from scipy import special
from scipy import stats


# --------------------------------------------------------------------------------
# continuous distributions
# --------------------------------------------------------------------------------
def normal(x, mu, sigma):
    """Normal distribution
    """
    coeff = 1.0 / (sigma * np.sqrt(2*np.pi))
    return coeff * np.exp(0.5 * ((x-mu)/sigma)**2)


def lognormal(x, mu, sigma):
    """Normal distribution
    """
    coeff = 1.0 / (x * sigma * np.sqrt(2*np.pi))
    return coeff * np.exp(0.5 * ((np.log(x)-mu)/sigma)**2)


def chi(x, k):
    return ( x**(k-1)*np.exp(-0.5 * x**2) ) / ( 2**(0.5*k-1)*special.gamma(0.5*k) )


def dagum(x, a, b, p):
    def cumdagum(x):
        return ( 1+(x/b)**-a )**-p

    # distance used to calculate the derivative
    dx = 1e-7 * np.min(np.abs(x))

    # calculate derivative of cumulative probability distribution
    prob = (cumdagum(x+dx) - cumdagum(x-dx)) / (2*dx)
    # # normalize probability distribution
    # prob = prob / np.sum(prob)
    
    return prob


# --------------------------------------------------------------------------------
# discrete distributions
# --------------------------------------------------------------------------------
def poisson(x, mu):
    """Continuous version of the Poisson distribution
    
    Parameters
    ----------
    x: array_like
        Value(s) used to sample the Poisson distribution.
        
    mu: float
        Mean of the Poisson distribution.
        
    Returns
    -------
    y: array_like
        Array with the same shape as x.  It contains the probability density
        for the given sampling x.
    """
    return mu**x * np.exp(-mu) / special.gamma(x+1)


def binom(x, n, p):
    """Continuous version of the Binomial distribution
    
    Parameters
    ----------
    x: array_like
        Value(s) used to sample the Binomial distribution.
        
    n: float
        Number of experiments
        
    p: float
        probability of that an experiment is successful.
        
    Returns
    -------
    y: array_like
        Array with the same shape as x.  It contains the probability density
        for the given sampling x.
    """
    coeff = special.gamma(n+1) / (special.gamma(x+1)*special.gamma(n-x+1))
    return coeff * p**x * (1-p)**(n-x)
