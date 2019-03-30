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

def train_one(data_name, n_pairs, data_config, CU_config, TaylorCU_config, random_seed):
    np.random.seed(random_seed)
    data_loader  = DataLoader(data_name, data_config)
    Compara_X, Compara_y = data_loader.build_comaparative_dataset(n_pairs)
    cdf_func, dense_func = get_CDF_and_dense(data_name, data_config, data_loader.y)

    mdl = SVMRank()
    mdl.fit(Compara_X,Compara_y,cdf_func,data_loader.X)
    min_y, max_y = np.min(data_loader.y), np.max(data_loader.y)
    err = data_loader.y - mdl.predict_unlabeled(lowerS = min_y, upperS = max_y)
    
    X_plus, X_minus = DataLoader.transform_dataset_to_CU(Compara_X, Compara_y)
    mdl1 = CUmodel()
    mdl1.fit(X_plus,X_minus,cdf_func,data_loader.X, **CU_config)
    err1 = data_loader.y - mdl1.predict_val(data_loader.X, lowerS = min_y, upperS = max_y)

    mdl2 = TaylorCUmodel()
    taylor_point = np.mean(data_loader.y)
    dense_at_taylor = dense_func(taylor_point)
    cdf_at_taylor = cdf_func(taylor_point)
    mdl2.fit(X_plus,X_minus, taylor_point, dense_at_taylor, cdf_at_taylor,data_loader.X, **TaylorCU_config)
    err2 = data_loader.y - mdl2.predict_val(data_loader.X)
    
    return np.array((np.mean(err*err),np.mean(err1*err1), np.mean(err2*err2)))


@click.command()
@click.argument("config_name")
@click.option('--nParallel', '-t', default=1)
def train_all(config_name, nparallel):
    with open(config_name, "r") as f:
        config = json.load(f)
    torch.set_num_threads(nparallel)
    data_name = config["data_name"]
    n_pairs = config["n_pairs"]
    data_config = config["data_config"]
    CU_config = config["CU_option"]
    TaylorCU_config = config["TaylorCU_option"]
    n_repeat = config["n_repeats"]
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)
    
    res = [train_one(data_name, n_pairs, data_config, CU_config, TaylorCU_config, i) for i in range(n_repeat)]
    res = np.array(res)
    np.save(config_name[:-5]+".res.npy", res)


def main():
    train_all()

if __name__ == "__main__":
    main()
