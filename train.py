import numpy as np
from DataLoader import DataLoader
from linear_svm import SVMRank
from CUmodel import CUmodel
from TaylorCUmodel import TaylorCUmodel
from utils import get_CDF_and_dense
from multiprocessing import Pool
import json
import click
import logging
import torch
from sklearn.linear_model import LinearRegression
logger = logging.getLogger(__name__)

n_models = 4

def train_one(data_name, n_pairs, data_config, CU_config, TaylorCU_config, random_seed):
    data_loader  = DataLoader(data_name, data_config, random_seed)
    Compara_X, Compara_y = data_loader.build_comaparative_dataset(n_pairs)
    cdf_func, dense_func, link_func = get_CDF_and_dense(data_name, data_config, data_loader.y)

    mdl = SVMRank()
    mdl.fit(Compara_X,Compara_y,cdf_func,data_loader.X)
    min_y, max_y = np.min(data_loader.y), np.max(data_loader.y)
    err = data_loader.y - mdl.predict_unlabeled(lowerS = min_y, upperS = max_y)


    X_plus, X_minus = DataLoader.transform_dataset_to_CU(Compara_X, Compara_y)
    mdl1 = CUmodel()
    mdl1.fit(X_plus,X_minus,cdf_func, link_func, data_loader.X,**CU_config)
    err1 = data_loader.y - mdl1.predict_val(data_loader.X, lowerS = min_y, upperS = max_y)


    mdl2 = TaylorCUmodel()
    taylor_point = np.mean(data_loader.y)
    dense_at_taylor = dense_func(taylor_point)
    cdf_at_taylor = cdf_func(taylor_point)
    mdl2.fit(X_plus,X_minus, taylor_point, dense_at_taylor, cdf_at_taylor,data_loader.X, **TaylorCU_config)
    err2 = data_loader.y - mdl2.predict_val(data_loader.X)

    lr = LinearRegression()
    lr.fit(data_loader.X, data_loader.y)
    err3 = data_loader.y - lr.predict(data_loader.X)

    logger.debug((np.mean(err*err),np.mean(err1*err1), np.mean(err2*err2), np.mean(err3*err3)))
    logger.debug(mdl1.param)
    logger.debug(mdl2.param)
    logger.debug(lr.coef_)
    return np.array((np.mean(err*err),np.mean(err1*err1), np.mean(err2*err2), np.mean(err3*err3)))


@click.command()
@click.argument("config_name")
@click.option('--nParallel', '-t', default=1)
def train_all(config_name, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)
    data_name = config["data_name"]
    n_pairs_list = config["n_pairs"]
    data_config = config["data_config"]
    CU_config = config["CU_option"]
    TaylorCU_config = config["TaylorCU_option"]
    n_repeat = config["n_repeats"]
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    res_means = np.empty((n_models, len(n_pairs_list)))
    res_vars = np.empty((n_models, len(n_pairs_list)))
    
    for idx, n_pairs in enumerate(n_pairs_list):
        res = [train_one(data_name, n_pairs, data_config, CU_config, TaylorCU_config, i) for i in range(n_repeat)]
        res = np.array(res)
        res_means[:,idx] = np.mean(res, axis = 0)
        res_vars[:, idx] = np.var(res, axis = 0)

    np.save(config_name[:-5]+".mean.npy", res_means)
    np.save(config_name[:-5]+".var.npy", res_vars)

def main():
    train_all()

if __name__ == "__main__":
    main()
