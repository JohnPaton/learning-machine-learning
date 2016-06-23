import numpy as np
import random

# simulate N data points using polynomial coefficients coefs with domain lsit xrange
# coefs[i] is coefficient of x**i
# x points regularly spaced if xrand is False
# returns np arrays x,y, sorted in increasing x order
def simData(coefs, N, xrange=[-1,1], xrand = True, noise=0.1):

    lower = xrange[0]
    upper = xrange[1]

    if not xrand:
        xs = np.linspace(lower,upper,N)
    else:
        xs = np.array([random.uniform(lower,upper) for _ in range(0,N)])
        xs = np.sort(xs)

    # np uses reverse coefficient order
    ys = np.polyval(list(reversed(coefs)),xs)

    # multiplicative noise
    ys = np.array([random.gauss(1, noise)*y for y in ys])

    # additive noise
    ys = np.array([random.gauss(0, noise*(max(ys) - min(ys))) + y for y in ys]);

    return xs,ys
