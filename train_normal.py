import numpy as np
from data_loader import load_normal_data, load_boston_data, load_unif_data, load_cos_data, load_boston_data
from CUmodel import CUmodel
from TaylorCUmodel import TaylorCUmodel
from linear_svm import SVMRank
from utils import normal_CDF, normal_dense, build_dataset, find_quantile_one, uniform_CDF, KDE_CDF, KDE_dense
from math import sin, cos, pi
from multiprocessing import Pool
import time


def train_TaylorCU_on_normal(total_n_sample, n_sample, nDim, noise = 0.1, use_simplified = True, use_unlabeled=False, reg = 0.0, rand_seed = 42):
    np.random.seed(rand_seed)
    theta = np.random.normal(0,1,nDim)
    theta /= np.linalg.norm(theta)
    X,y = load_normal_data(total_n_sample, nDim, theta, 0.1)
    cdf_func = normal_CDF(0.0, np.sqrt(1.0 + 0.01))
    dense_func = normal_dense(0.0, np.sqrt(1.0 + 0.01))
    taylor_point = 0.0
    
    mdl = TaylorCUmodel()
    mdl.fit(X, y, taylor_point, dense_func(taylor_point), cdf_func(taylor_point), n_sample, use_unlabeled, rand_seed = rand_seed, reg = reg)
    err = y - mdl.predict_val(X)
    
    mdl1 = CUmodel()
    mdl1.fit(X,y,cdf_func, n_sample, use_unlabeled,use_simplified, rand_seed = rand_seed, reg = reg)
    err1 = y - mdl1.predict_val(X)


    mdl2 = SVMRank()
    mdl2.fit(X,y,cdf_func,n_sample,use_unlabeled=True)
    err2 = y - mdl2.predict_unlabeled()
    return np.mean(err*err), np.mean(err1*err1), np.mean(err2*err2)
    

def train_one(n_sample):
    total_n_sample = 1000000
    res = [train_TaylorCU_on_normal(total_n_sample, n_sample, 5, 0.1, True, True, 0.0, i) for i in range(100)]
    res = np.array(res)
    np.save("%s_res.npy"%n_sample, res)
    

if __name__ == "__main__":
    train_one(10)
    train_one(100)
    train_one(1000)
    train_one(10000)
        
