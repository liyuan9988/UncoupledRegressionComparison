import numpy as np
from DataLoader import DataLoader
from linear_svm import SVMRank
from CUmodel import CUmodel
from TaylorCUmodel import TaylorCUmodel
from OptimizedTaylorCUmodel import OptimizedTaylorCUmodel
from linear_shuffle import LinearShuffle
from utils import get_CDF_and_dense
from multiprocessing import Pool
import json
import click
import logging
import torch
from sklearn.linear_model import LinearRegression
logger = logging.getLogger(__name__)

n_models = 6


def train_one(data_name, n_pairs, data_config, CU_config, TaylorCU_config, random_seed):
    data_loader = DataLoader(data_name, data_config, random_seed)
    Compara_X, Compara_y = data_loader.build_comaparative_dataset(data_loader.X_train, data_loader.y_train, n_pairs)
    cdf_func, dense_func, link_func = get_CDF_and_dense(data_name, data_config, data_loader.y_train)
    min_y = min(np.min(data_loader.y_test), np.min(data_loader.y_train))
    max_y = max(np.max(data_loader.y_test), np.max(data_loader.y_train))

    mdl0 = SVMRank()
    mdl0.fit(Compara_X, Compara_y, cdf_func, data_loader.X_train, loss="hinge")
    err0 = data_loader.y_test - mdl0.predict_val(data_loader.X_test, lowerS=min_y, upperS=max_y)

    #mdl = SVMRank()
    #mdl.fit(Compara_X,Compara_y,cdf_func,data_loader.X_train, loss="squared_hinge")
    #err = data_loader.y_test - mdl.predict_val(data_loader.X_test, lowerS = min_y, upperS = max_y)
    err = 0.0

    X_plus, X_minus = DataLoader.transform_dataset_to_CU(Compara_X, Compara_y)
    mdl1 = CUmodel()
    mdl1.fit(X_plus, X_minus, cdf_func, link_func, data_loader.X_train, **CU_config)
    err1 = data_loader.y_test - mdl1.predict_val(data_loader.X_test, lowerS=min_y, upperS=max_y)

    logger.debug("train LR")
    lr = LinearRegression()
    lr.fit(data_loader.X_train, data_loader.y_train)
    err3 = data_loader.y_test - lr.predict(data_loader.X_test)
    logger.debug("end LR")

    mdl3 = OptimizedTaylorCUmodel()
    optimal_weight = None
    if(data_name == "uniform"):
        optimal_weight = np.array([0.5, 0.5, -0.5])
    mdl3.fit(X_plus, X_minus, cdf_func, dense_func, min_y, max_y, data_loader.X_train, optimal_weight)
    err4 = data_loader.y_test - mdl3.predict_val(data_loader.X_test)

    mdl4 = LinearShuffle()
    mdl4.fit(data_loader.X_train, data_loader.y_train)
    err5 = data_loader.y_test - mdl4.predict_val(data_loader.X_test)

    logger.info((np.mean(err0*err0), np.mean(err1*err1), np.mean(err3*err3), np.mean(err4*err4), np.mean(err5*err5)))
    logger.debug(mdl1.param)
    logger.debug(mdl3.param)
    logger.debug(lr.coef_)
    return np.array((np.mean(err0*err0), np.mean(err*err), np.mean(err1*err1), np.mean(err3*err3), np.mean(err4*err4), np.mean(err5*err5)))


@click.command()
@click.argument("config_name")
@click.option('--nParallel', '-t', default=1)
def train_all(config_name, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)
    data_name_list = config["data_name"]
    n_pairs_list = config["n_pairs"]
    data_config = config["data_config"]
    CU_config = config["CU_option"]
    TaylorCU_config = config["TaylorCU_option"]
    n_repeat = config["n_repeats"]
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    res_means = np.empty((n_models, len(n_pairs_list), len(data_name_list)))
    res_vars = np.empty((n_models, len(n_pairs_list), len(data_name_list)))

    for idx2, data_name in enumerate(data_name_list):
        for idx1, n_pairs in enumerate(n_pairs_list):
            res = [train_one(data_name, n_pairs, data_config, CU_config, TaylorCU_config, i) for i in range(n_repeat)]
            res = np.array(res)
            res_means[:, idx1, idx2] = np.mean(res, axis=0)
            res_vars[:, idx1, idx2] = np.var(res, axis=0)

    np.save(config_name[:-5]+".mean.npy", res_means)
    np.save(config_name[:-5]+".var.npy", res_vars)


if __name__ == "__main__":
    train_all()
