import numpy as np
import random
import matplotlib.pyplot as plt

# simulate N data points using polynomial coefficients coefs with domain lsit xrange
# coefs[i] is coefficient of x**i
# x points regularly spaced if xrand is False
# returns np arrays x,y, sorted in increasing x order
def simPoly(coefs, N, xrange=[-1,1], xrand = True, noise=0.1):

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

# simulate N culustered datapoints
# centroids is list of coords of centroids (these also set the dimension)
# spreads is list of sd. dev. of clusters (single int to make them all the same)
# labels is list of labels of points from different centroids (default '0','1',...)
# returns lists of coords [xs, ys, ... , labels]
def simClust(centroids, N, spreads = 1, labels = []):
    ncent= len(centroids)
    dim = len(centroids[0])

    # set up default labels
    if not labels:
        labels = [str(i) for i in range(ncent)]

    # set up spreads of gaussians
    if type(spreads) is int or type(spreads) is float:
        sp = [spreads for _ in range(ncent)]
    else:
        sp = spreads

    # simulate data
    data = [[] for _ in range(dim+1)]
    for p in range(N):
        cent = random.randint(0,ncent-1)
        for i in range(dim):
            x = random.gauss(centroids[cent][i], sp[cent])
            data[i].append(x)

        data[i+1].append(labels[cent])

    return data
            

data = simClust([[0,0],[5,5],[1,8]],1000, spreads = [1,2,1],labels=['r','k','b'])
xs = data[0]
ys = data[1]

plt.scatter(xs,ys,c=data[2])
plt.show()

    

    
    
