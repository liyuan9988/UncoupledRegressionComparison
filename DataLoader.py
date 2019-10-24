import numpy as np
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from math import pi, cos
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self, data_name, data_config, random_seed = 42):
        self.data_name = data_name
        self.data_config = data_config
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.refresh_data()
    
    def refresh_data(self):
        logger.info("start building data: %s"%self.data_name)
        if(self.data_name == "uniform"):
            self.load_unif_data(**self.data_config)
        elif(self.data_name == "normal"):
            self.load_normal_data(**self.data_config)
        elif(self.data_name == "boston"):
            self.load_boston_data(**self.data_config)
        elif(self.data_name == "diabetes"):
            self.load_diabetes_data(**self.data_config)
        elif(self.data_name == "airfoil"):
            self.load_airfoil_data(**self.data_config)
        elif(self.data_name == "concrete"):
            self.load_concrete_data(**self.data_config)    
        elif(self.data_name == "power"):
            self.load_power_data(**self.data_config)
        elif(self.data_name == "mpg"):
            self.load_mpg_data(**self.data_config)
        elif(self.data_name == "crime"):
            self.load_crime_data(**self.data_config)    
        elif(self.data_name == "cbm"):
            self.load_cbm_data(**self.data_config)
        elif(self.data_name == "grid"):
            self.load_grid_data(**self.data_config)
        elif(self.data_name == "redwine"):
            self.load_redwine_data(**self.data_config)
        elif(self.data_name == "whitewine"):
            self.load_whitewine_data(**self.data_config)   
        elif(self.data_name == "abalone"):
            self.load_abalon_data(**self.data_config)
        elif(self.data_name == "cpu"):
            self.load_cpu_data(**self.data_config)
        else:
            raise ValueError("invalid data name %s"%self.data_name)
        logger.info("complete building")

    @staticmethod
    def build_comaparative_dataset(X, y, n_pairs):
        logger.info("start tranform comparative")
        nData, nDim = X.shape
        Compara_X = np.empty((n_pairs, nDim*2))
        Compara_y = np.empty((n_pairs,))
        for i in range(n_pairs):
            idx1,idx2 = np.random.choice(nData, 2, False)
            Compara_X[i,:nDim] = X[idx1]
            Compara_X[i,nDim:] = X[idx2]
            if(y[idx1] >= y[idx2]):
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
            if(Compara_y[i] >= 0.5):
                X_plus[i] = Compara_X[i,:nDim]
                X_minus[i] = Compara_X[i,nDim:]
            else:
                X_plus[i] = Compara_X[i,nDim:]
                X_minus[i] = Compara_X[i,:nDim]
        logger.info("end transform CU")
        return X_plus, X_minus

    def load_unif_data(self, n_sample, nDim, const = 1.0, lower = 0.0, upper = 1.0, noise = 0.1, test_ratio=0.2, **kwargs):
        self.X_train = np.random.uniform(lower,upper,(n_sample,nDim))
        ntest = int(test_ratio*n_sample)
        self.X_test = np.random.uniform(lower,upper,(ntest,nDim))
        theta = np.zeros(nDim)
        theta[0] = const
        self.y_train = self.X_train.dot(theta) + np.random.normal(0.0,noise,n_sample)
        self.y_test = self.X_test.dot(theta) + np.random.normal(0.0,noise,ntest)
        

    def load_normal_data(self, n_sample, nDim, noise = 0.1, test_ratio=0.2, **kwargs):
        theta = np.random.normal(0,1,nDim)
        theta /= np.linalg.norm(theta)
        if(nDim == 1):
            theta = np.array([1.0])
        self.X_train = np.random.normal(0,1,(n_sample,nDim))
        ntest = int(test_ratio*n_sample)
        self.X_test = np.random.normal(0,1,(ntest,nDim))
        self.y_train = self.X_train.dot(theta) + np.random.normal(0,noise,n_sample)
        self.y_test = self.X_test.dot(theta) + np.random.normal(0,noise,ntest)


    def load_boston_data(self, test_ratio=0.2, **kwargs):
        boston = load_boston()
        X = np.array(boston.data, copy=True)
        y = np.array(boston.target, copy=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)
        

    def load_diabetes_data(self, test_ratio=0.2, **kwargs):
        diabetes = load_diabetes()
        X = np.array(diabetes.data, copy=True)
        y = np.array(diabetes.target, copy=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_airfoil_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/airfoil_self_noise.dat")
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_concrete_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/Concrete_Data.csv")
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_power_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/power_plant.csv")
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_mpg_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/auto-mpg.csv")
        data.dropna(inplace=True)
        X = data.values[:,1:]
        y = data.values[:,0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)
    
    def load_crime_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/communities.data", header=None).iloc[:,5:]
        data.dropna(inplace=True)
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)
    
    def load_cbm_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/cbm_data.txt", header=None).iloc[:,1:-1]
        data.dropna(inplace=True)
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_grid_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/grid_stability.csv").iloc[:,:-1]
        data.dropna(inplace=True)
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)
    
    def load_redwine_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/winequality-red.csv",delimiter=";")
        data.dropna(inplace=True)
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)
    
    def load_whitewine_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/winequality-white.csv",delimiter=";")
        data.dropna(inplace=True)
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_abalon_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/abalone.data", header=None)
        data.dropna(inplace=True)
        X = pd.get_dummies(data.iloc[:,:-1]).values
        y = data.iloc[:,-1].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)

    def load_cpu_data(self, test_ratio=0.2, **kwargs):
        data = pd.read_csv("data/machine.data", header=None)
        data.dropna(inplace=True)
        X = data.values[:,:-1]
        y = data.values[:,-1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=self.random_seed)


        
