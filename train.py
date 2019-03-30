import numpy as np
from DataLoader import DataLoader
from linear_svm import SVMRank
from utils import normal_CDF

def train(data_name, data_config):
    data_loader  = DataLoader(data_name, data_config)
    Compara_X, Compara_y = data_loader.build_comaparative_dataset(1000)
    cdf_func = normal_CDF(0.0, np.sqrt(1.0 + 0.01))
    mdl = SVMRank()
    mdl.fit(Compara_X,Compara_y,cdf_func,data_loader.X)
    min_y, max_y = np.min(data_loader.y), np.max(data_loader.y)
    err = data_loader.y - mdl.predict_unlabeled(lowerS = min_y, upperS = max_y)
    print(err)


if __name__ == "__main__":
   import logging
   logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
   data_name = "normal"
   data_config = dict(n_sample = 1000, nDim = 5, noise = 0.1)    
   train(data_name, data_config)