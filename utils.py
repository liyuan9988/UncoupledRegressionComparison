import numpy as np
from scipy.special import erfinv, erf
from math import pi
import torch
from scipy.special import expit
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import logging
logger = logging.getLogger(__name__)



def find_quantile_one(cdf_func, prob, lowerS = -10.0, upperS = 10.0):
    assert len(prob.shape) == 1
    prob[prob > 0.999] = 0.999
    prob[prob < 0.001] = 0.001
    nData = len(prob)
    lower_p = np.ones(nData) * cdf_func(lowerS)
    upper_p = np.ones(nData) * cdf_func(upperS)
    lowerS = np.ones(nData) * lowerS
    upperS = np.ones(nData) * upperS
    while(np.max(upper_p - lower_p) > 1.0e-5):
        mid_S = (lowerS + upperS)/2
        mid_p = cdf_func(mid_S)
        upper_p[mid_p > prob] = mid_p[mid_p > prob]
        upperS[mid_p > prob] = mid_S[mid_p > prob]
        lower_p[mid_p < prob] = mid_p[mid_p < prob]
        lowerS[mid_p < prob] = mid_S[mid_p < prob]
        lower_p[upperS-lowerS < 1.0e-5] = upper_p[upperS-lowerS < 1.0e-5]
    return lowerS

def obtain_optimal_sigma(y_array):
    grid = GridSearchCV(KernelDensity(kernel="gaussian"),
                        {'bandwidth': np.linspace(0.1, 20, 10)},
                        cv=5)
    grid.fit(y_array[:,None])
    return grid.best_params_["bandwidth"]

def get_CDF_and_dense(data_name, data_config, y_array):
    if(data_name == "uniform"):
        const = data_config.get("const", 1.0)
        lower = data_config.get("lower", 0.0) * const
        upper = data_config.get("upper", 1.0) * const
        cdf_func = uniform_CDF(min(lower,upper), max(lower,upper))
        dense_func = lambda x: 1.0
        link_func = cdf_func
    elif(data_name == "normal"):
        mu = 0.0
        noise = data_config.get("noise", 0.1)
        var = np.sqrt(1.0 + noise*noise)
        cdf_func = normal_CDF(mu, var)
        dense_func = normal_dense(mu, var)
        link_func = expit
    else:
        if(len(y_array) > 2000):
            small_y_array = np.random.choice(y_array,2000)
        else:
            small_y_array = y_array
        logger.debug(len(small_y_array))
        sigma = obtain_optimal_sigma(small_y_array)
        logger.debug("tuned_sigma:%f"%sigma)
        cdf_func = KDE_CDF(small_y_array, sigma)
        #cdf_func = empi_CDF(y_array)
        dense_func = KDE_dense(small_y_array, sigma)
        link_func = expit

    return cdf_func, dense_func, link_func
        



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
        return np.exp(-z*z/2.0)/np.sqrt(2*pi) 
    return standard

def KDE_CDF(y_array, sigma = 0.1):
    def standard(x,mu,var):
        z = (x-mu)/var
        return (1 + erf(z/np.sqrt(2)))/2.0
    
    return lambda x: np.mean(np.array([standard(x,y,sigma) for y in y_array]), axis = 0)


def KDE_dense(y_array, sigma = 0.1):
    def standard(x, mu, var):
        z = (x-mu)/sigma
        return np.exp(-z*z/2.0)/np.sqrt(2*pi) 
        
    return lambda x: np.mean(np.array([standard(x,y,sigma) for y in y_array]), axis = 0)

def empi_CDF(y_array):
    return np.vectorize(lambda x: (1+np.sum(y_array < x)/len(y_array)))
