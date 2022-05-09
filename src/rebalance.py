import numpy as np
from scipy import optimize

def f(x, w_target, value, w, c):
    return w_target*(value+c) - (x + w*value)

def g(x, w_target, value, w, c):
    return np.sum(f(x, w_target, value, w, c)**2)

def optimal_rebalance_while_buying(value, c, w, w_target):
    """
    Rebalance the portfolio with contributions, without selling stocks. An optimal allocation of the contribution is
    found via numerical optimization that matches as closely as possible the target allocation, without having to sell
    stocks.

    NAME: value
    DESC: Portfolio value.
    TYPE: Float

    NAME: c
    DESC: contribution amount.
    TYPE: Float

    NAME: w
    DESC: Current portfolio allocation: array of market value percentages.
    TYPE: Numpy array

    NAME: w_target
    DESC: Target portfolio allocation: array of market value percentages.
    TYPE: Numpy array
    """
    n_shares = np.size(w)
    x0 = np.full((1, n_shares), c * 1 / n_shares).transpose()
    bnds = [(0,c) for i in range(0,n_shares)]
    constraint = optimize.LinearConstraint(np.ones(n_shares), c, c, keep_feasible=False)
    res = optimize.minimize(g, x0, method="SLSQP", bounds=bnds, constraints=constraint, args=(w_target, value, w, c,),tol=1e-6)
    return res['x']

def standard_rebalance(value, c, w, w_target):
    """
    Rebalance the portfolio selling and buying stocks.

    NAME: value
    DESC: Portfolio value.
    TYPE: Float

    NAME: c
    DESC: contribution amount.
    TYPE: Float

    NAME: w
    DESC: Current portfolio allocation: array of market value percentages.
    TYPE: Numpy array

    NAME: w_target
    DESC: Target portfolio allocation: array of market value percentages.
    TYPE: Numpy array
    """
    n_shares = np.size(w)
    new_alloc = (c+value)*w_target
    old_alloc = value*w
    return new_alloc-old_alloc

# w_target = np.array([0.35, 0.17, 0.16, 0.12, 0.085, 0.115])
# w = np.array([0.328381731865964, 0.170038379728105, 0.151010357902525, 0.121975091710595, 0.11766315237831, 0.110931286414501])
# value = 284362 # sum of market value
# c = 24850
# x = optimal_rebalance_while_buying(value, c, w, w_target)