import numpy as np
from sklearn.datasets import load_boston
from math import pi, cos
np.random.seed(42)
import logging
logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self, data_name, data_config, random_seed = 42):
        self.data_name = data_name
        self.data_config = data_config
        np.random.seed(random_seed)
        self.refresh_data()
    
    def refresh_data(self):
        logger.info("start building data: %s"%self.data_name)
        if(self.data_name == "uniform"):
            self.X, self.y = self.load_unif_data(**self.data_config)
        elif(self.data_name == "normal"):
            self.X, self.y = self.load_normal_data(**self.data_config)
        elif(self.data_name == "cos"):
            self.X, self.y = self.load_cos_data(**self.data_config)
        elif(self.data_name == "boston"):
            self.X, self.y = self.load_boston_data()
        else:
            raise ValueError("invalid data name %s"%self.data_name)
        logger.info("complete building")

    def build_comaparative_dataset(self, n_pairs):
        logger.info("start tranform comparative")
        nData, nDim = self.X.shape
        Compara_X = np.empty((n_pairs, nDim*2))
        Compara_y = np.empty((n_pairs,))
        for i in range(n_pairs):
            idx1,idx2 = np.random.choice(nData, 2, False)
            Compara_X[i,:nDim] = self.X[idx1]
            Compara_X[i,nDim:] = self.X[idx2]
            if(self.y[idx1] >= self.y[idx2]):
                Compara_y[i] = 1
            else:
                Compara_y[i] = 0
        logger.info("end tranform comparative")
        return Compara_X, Compara_y

    @staticmethod
    def transform_dataset_to_CU(Compara_X, Compara_y):
        logger.info("start transform CU")
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
        logger.info("end transform CU")
        return X_plus, X_minus

    def load_unif_data(self, n_sample, nDim, const, lower = 0.0, upper = 1.0, noise = 0.1):
        X = np.random.uniform(lower,upper,(n_sample,nDim))
        theta = np.zeros(nDim)
        theta[0] = const
        Y = X.dot(theta) + np.random.normal(0.0,noise,n_sample)
        return (X,Y)

    def load_normal_data(self, n_sample, nDim, noise = 0.1):
        theta = np.random.normal(0,1,nDim)
        theta /= np.linalg.norm(theta)
        X = np.random.normal(0,1,(n_sample,nDim))
        y = X.dot(theta) + np.random.normal(0,noise,n_sample)
        return (X, y)


    def load_cos_data(self, n_sample, nDim, const, noise = 0.1):
        X = np.random.uniform(0.0, 1.0, (n_sample,nDim))
        X[:,0] = np.arcsin(X[:,0])
        Y = X[:,0]*const + np.random.normal(0.0,noise,n_sample)        
        return X, Y

    def load_boston_data(self):
        boston = load_boston()
        X = np.array(boston.data, copy=True)
        y = np.array(boston.target, copy=True)
        return X,y

