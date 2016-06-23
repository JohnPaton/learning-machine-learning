from statistics import mean
import numpy as np
import matplotlib.pyplot as plt

# find the best fit slope for a list of xs and ys
def fit_slope(xs,ys):
    cov = mean(xs*ys) - mean(xs)*mean(ys)
    var = mean(xs**2) - mean(xs)**2
    m = cov/var
    return m

# find m and b for the best fit line ys = m * xs + b
def best_fit(xs,ys):
    m = fit_slope(xs,ys)
    b = mean(ys) - m * mean(xs)
    return m,b

# find the coefficient vector beta for an array of features Xs and labels ys
def fit_multi(Xs,ys):

    # column of 1s for intercept
    n = np.ma.size(Xs,0) # number of data points
    X = np.insert(Xs,0,np.ones(n),axis=1)
    
    XT = np.matrix.transpose(X)
    XTX = np.dot(XT,X)
    XTXinv = np.linalg.inv(XTX)
    XTY = np.dot(XT,ys)

    beta = np.dot(XTXinv,XTY)

    return beta

# make a prediction using features Xs and linear model coefficients beta
def predict_multi(Xs,beta):

    # column of 1s for intercept
    n = np.ma.size(Xs,0) # number of data points
    X = np.insert(Xs,0,np.ones(n),axis=1)

    y = np.dot(X,beta)

    return y
    

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def r2(ys_orig, ys_line):
    y_mean_line = mean(ys_orig)*np.ones(len(ys_orig))
    reg_err = squared_error(ys_orig, ys_line)
    mean_err = squared_error(ys_orig, y_mean_line);

    return 1-(reg_err/mean_err)
    

