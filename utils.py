import numpy as np
from scipy.special import erfinv, erf
from math import pi

def build_dataset(org_X, org_y, n_sample, rand_seed = 42):
    np.random.seed(rand_seed)
    nData, nDim = org_X.shape
    Compara_X = np.empty((n_sample, nDim*2))
    Compara_y = np.empty((n_sample,))
    for i in range(n_sample):
        idx1,idx2 = np.random.choice(nData, 2, False)
        Compara_X[i,:nDim] = org_X[idx1]
        Compara_X[i,nDim:] = org_X[idx2]
        if(org_y[idx1] >= org_y[idx2]):
            Compara_y[i] = 1
        else:
            Compara_y[i] = 0
    return Compara_X, Compara_y


def transform_dataset_to_CU(Compara_X, Compara_y):
    n_sample, nDim = Compara_X.shape[0], int(Compara_X.shape[1] / 2 )
    X_plus = np.empty((n_sample, nDim))
    X_minus = np.empty((n_sample, nDim))
    for i in range(n_sample):
        if(Compara_y[i] == 1):
            X_plus[i] = Compara_X[i,:nDim]
            X_minus[i] = Compara_X[i,nDim:]
        else:
            X_plus[i] = Compara_X[i,nDim:]
            X_minus[i] = Compara_X[i,:nDim]
    return X_plus, X_minus

def find_quantile_one(cdf_func, prob, lowerS = -10.0, upperS = 10.0):
    lower_p = cdf_func(lowerS)
    upper_p = cdf_func(upperS)
    if(lower_p > prob):
        return lowerS
    elif(upper_p < prob):
        return upperS
    else:
        while(upper_p - lower_p > 1.0e-5):
            mid_S = (lowerS + upperS)/2
            mid_p = cdf_func(mid_S)
            if(mid_p > prob):
                upper_p = mid_p
                upperS = mid_S
            else:
                lower_p = mid_p
                lowerS = mid_S
            #print(lower_p, upper_p)
        return mid_S

def uniform_CDF(lower=0.0, upper = 1.0):
    def cdf(x):
        res = (x - lower)/(upper - lower)
        return np.where(res < 0.0, 0.0, np.where(res > 1.0, 1.0, res))
    return cdf 
    
def normal_CDF(mu=0.0, sigma = 1.0):
    def standard(x):
        z = (x-mu)/sigma
        return (1 + erf(z/np.sqrt(2)))/2.0
    return standard

def normal_dense(mu=0.0, sigma = 1.0):
    def standard(x):
        z = (x-mu)/sigma
        return np.exp(-z*z)/np.sqrt(2*pi) 
    return standard

def KDE_CDF(y_array, sigma = 0.1):
    def standard(x,mu,var):
        z = (x-mu)/var
        return (1 + erf(z/np.sqrt(2)))/2.0
    
    return lambda x: np.mean(np.array([standard(x,y,sigma) for y in y_array]), axis = 0)


def KDE_dense(y_array, sigma = 0.1):
    def standard(x, mu, var):
        z = (x-mu)/sigma
        return np.exp(-z*z)/np.sqrt(2*pi) 
        
    return lambda x: np.mean(np.array([standard(x,y,sigma) for y in y_array]), axis = 0)
