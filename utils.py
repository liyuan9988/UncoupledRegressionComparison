import numpy as np
from scipy.special import erfinv, erf
from math import pi

def find_quantile_one(cdf_func, prob, lowerS = -10.0, upperS = 10.0):
    assert len(prob.shape) == 1
    nData = len(prob)
    lower_p = np.ones(nData) * cdf_func(lowerS)
    upper_p = np.ones(nData) * cdf_func(upperS)
    lowerS = np.ones(nData) * lowerS
    upperS = np.ones(nData) * upperS
    while(np.min(upper_p - lower_p) > 1.0e-5):
        mid_S = (lowerS + upperS)/2
        mid_p = cdf_func(mid_S)
        upper_p[mid_p > prob] = mid_p[mid_p > prob]
        upperS[mid_p > prob] = mid_S[mid_p > prob]
        lower_p[mid_p < prob] = mid_p[mid_p < prob]
        lowerS[mid_p < prob] = mid_S[mid_p < prob]
    return mid_S

def get_CDF_and_dense(data_name, data_config, y_array):
    if(data_name == "uniform"):
        lower = data_config.get("lower", 0.0) * data_config["const"]
        upper = data_config.get("upper", 1.0) * data_config["const"]
        cdf_func = uniform_CDF(lower, upper)
        dense_func = lambda x: 1.0
    elif(data_name == "normal"):
        mu = 0.0
        noise = data_config.get("noise", 0.1)
        var = np.sqrt(1.0 + noise*noise)
        cdf_func = normal_CDF(mu, var)
        dense_func = normal_dense(mu, var)
    else:
        cdf_func = KDE_CDF(y_array)
        dense_func = KDE_dense(y_array)

    return cdf_func, dense_func
        



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
