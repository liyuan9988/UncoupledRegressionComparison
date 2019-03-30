import numpy as np
from data_loader import load_normal_data, load_boston_data, load_unif_data, load_cos_data, load_boston_data
from CUmodel import CUmodel
from TaylorCUmodel import TaylorCUmodel
from linear_svm import SVMRank
from utils import normal_CDF, normal_dense, build_dataset, find_quantile_one, uniform_CDF, KDE_CDF, KDE_dense
from math import sin, cos, pi
from multiprocessing import Pool

def train_TaylorCU_on_boston(total_n_sample, n_sample, nDim, noise = 0.1, use_simplified = True, use_unlabeled=True, reg = 0.0):
    X,y = load_boston_data()
    print(X)
    max_y = np.max(y)
    min_y = np.min(y)
    cdf_func = KDE_CDF(y, sigma = 5.0)
    dense_func = KDE_dense(y, sigma = 5.0)

    taylor_point = np.mean(y)
    print(taylor_point)
    print(dense_func(taylor_point))
    print(cdf_func(taylor_point))
    mdl1 = TaylorCUmodel()
    mdl1.fit(X, y, taylor_point, dense_func(taylor_point), cdf_func(taylor_point), n_sample, use_unlabeled = True, rand_seed = 42, reg = 0.0)
    err1 = y - mdl1.predict_val(X)

    mdl2 = CUmodel()
    mdl2.fit(X,y,cdf_func, n_sample, use_unlabeled,use_simplified, 42, reg=0.0)
    err2 = y - mdl2.predict_val(X,lowerS = min_y, upperS = max_y)

    mdl3 = SVMRank()
    mdl3.fit(X,y,cdf_func,n_sample,use_unlabeled=True)
    err3 = y - mdl3.predict_unlabeled(lowerS = min_y, upperS = max_y)
    return np.mean(err1*err1), np.mean(err2*err2) , np.mean(err3*err3)




def train_TaylorCU_on_cos(total_n_sample, n_sample, nDim, noise = 0.1, use_simplified = True, use_unlabeled=True, reg = 0.0):
    X,y = load_cos_data(total_n_sample, nDim, const = 1.0, noise = 0.1)
    print(X)
    print(np.max(y))
    cdf_func = np.sin
    dense_func = np.cos

    taylor_point = 0.0
    mdl1 = TaylorCUmodel()
    mdl1.fit(X, y, taylor_point, dense_func(taylor_point), cdf_func(taylor_point), n_sample, use_unlabeled = True, rand_seed = 42, reg = 0.0)
    err1 = y - mdl1.predict_val(X)

    mdl2 = CUmodel()
    mdl2.fit(X,y,cdf_func, n_sample, use_unlabeled,use_simplified, 42, reg=0.0)
    err2 = y - mdl2.predict_val(X,lowerS = 0.0, upperS = pi/2.0)

    #mdl3 = SVMRank()
    #mdl3.fit(X,y,cdf_func,n_sample,use_unlabeled=True)
    #err3 = y - mdl3.predict_unlabeled(lowerS = 0.0, upperS = pi/2.0)
    return np.mean(err1*err1), np.mean(err2*err2) #, np.mean(err3*err3)




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


def train_CU_on_normal(total_n_sample, n_sample, nDim, noise = 0.1, use_simplified = True, use_unlabeled=False, reg = 0.0, rand_seed = 42):
    theta = np.random.normal(0,1,nDim)
    theta /= np.linalg.norm(theta)
    X,y = load_normal_data(total_n_sample, nDim, theta, 0.1)
    cdf_func = normal_CDF(0.0, np.sqrt(1.0))
    mdl = CUmodel()
    mdl.fit(X,y,cdf_func, n_sample, use_unlabeled,use_simplified, 42, reg=0.0)
    err = y - mdl.predict_val(X)

    mdl2 = SVMRank()
    mdl2.fit(X,y,cdf_func,n_sample,use_unlabeled=True)
    err1 = y - mdl2.predict_unlabeled()
    return np.mean(err*err), np.mean(err1*err1)


def train_CU_on_unif(total_n_sample, n_sample, nDim, upper=1.0, lower=0.0, const=2.0,noise = 0.1, use_simplified = True, use_unlabeled=False, reg = 0.0):
    X,y = load_unif_data(total_n_sample, nDim, const, lower, upper , noise)
    cdf_func = uniform_CDF(lower=lower*const, upper=upper*const)
    mdl = CUmodel()
    mdl.fit(X,y,cdf_func, n_sample, use_unlabeled,use_simplified, 42, reg=0.0)
    err = y - mdl.predict_val(X)

    mdl2 = SVMRank()
    mdl2.fit(X,y,cdf_func,n_sample,use_unlabeled=True)
    err1 = y - mdl2.predict_unlabeled()
    return np.mean(err*err), np.mean(err1*err1)


def train_one(n_sample):
    total_n_sample = 1000
    with Pool(processes=10) as p:
        res = p.starmap(train_TaylorCU_on_normal, [(total_n_sample, n_sample, 5, 0.1, True, True, 0.0, i) for i in range(100)])
        res = np.array(res)
    np.save("%s_res.npy"%n_sample, res)
    

if __name__ == "__main__":
    train_one(10)
    train_one(100)
    train_one(1000)
    train_one(10000)
        
