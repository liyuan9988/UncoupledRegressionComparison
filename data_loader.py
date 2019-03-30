import numpy as np
from sklearn.datasets import load_boston
from math import pi, cos
np.random.seed(42)

def load_normal_data(n_sample, nDim, theta, noise = 0.1):
    X = np.random.normal(0,1,(n_sample,nDim))
    y = X.dot(theta) + np.random.normal(0,noise,n_sample)
    return (X, y)

def load_unif_data(n_sample, nDim, const, lower = 0.0, upper = 1.0, noise = 0.1):
    X = np.random.uniform(lower,upper,(n_sample,nDim))
    theta = np.zeros(nDim)
    theta[0] = const
    Y = X.dot(theta) + np.random.normal(0.0,noise,n_sample)
    return (X,Y)

def load_cos_data(n_sample, nDim, const, noise = 0.1):
    X = np.random.uniform(0.0, 1.0, (n_sample,nDim))
    X[:,0] = np.arcsin(X[:,0])
    Y = X[:,0]*const + np.random.normal(0.0,noise,n_sample)        
    return X, Y

def load_boston_data():
    boston = load_boston()
    X = np.array(boston.data, copy=True)
    y = np.array(boston.target, copy=True)
    return X,y

