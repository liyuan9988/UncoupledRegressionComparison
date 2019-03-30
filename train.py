import numpy as np
from DataLoader import DataLoader
from linear_svm import SVMRank
from CUmodel import CUmodel
from TaylorCUmodel import TaylorCUmodel
from utils import normal_CDF, normal_dense

def train(data_name, data_config):
    data_loader  = DataLoader(data_name, data_config)
    Compara_X, Compara_y = data_loader.build_comaparative_dataset(10)
    cdf_func = normal_CDF(0.0, np.sqrt(1.0 + 0.01))
    mdl = SVMRank()
    mdl.fit(Compara_X,Compara_y,cdf_func,data_loader.X)
    min_y, max_y = np.min(data_loader.y), np.max(data_loader.y)
    err = data_loader.y - mdl.predict_unlabeled(lowerS = min_y, upperS = max_y)
    print(np.mean(err*err))

    X_plus, X_minus = DataLoader.transform_dataset_to_CU(Compara_X, Compara_y)
    mdl1 = CUmodel()
    mdl1.fit(X_plus,X_minus,cdf_func,data_loader.X, "sigmoid")
    err1 = data_loader.y - mdl1.predict_val(data_loader.X, lowerS = min_y, upperS = max_y)
    print(np.mean(err1*err1))

    mdl2 = TaylorCUmodel()
    dense_func = normal_dense(0.0, np.sqrt(1.0 + 0.01))
    taylor_point = np.mean(data_loader.y)
    dense_at_taylor = dense_func(taylor_point)
    cdf_at_taylor = cdf_func(taylor_point)
    mdl2.fit(X_plus,X_minus, taylor_point, dense_at_taylor, cdf_at_taylor,data_loader.X)
    err2 = data_loader.y - mdl2.predict_val(data_loader.X)
    print(np.mean(err2*err2))


if __name__ == "__main__":
   import logging
   logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s :%(message)s")
   data_name = "normal"
   data_config = dict(n_sample = 100000, nDim = 5, noise = 0.1)    
   train(data_name, data_config)