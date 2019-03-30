import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from utils import find_quantile_one, build_dataset, transform_dataset_to_CU

def score(x, mdl):
    return mdl._score(x)

class CUmodel:
    def __init__(self):
        pass
    
    def _score(self, param):
        unlabeled_predict = self.link_func(self.X_unlabeled.dot(param[1:]) + param[0])
        plus_predict = self.link_func(self.X_plus.dot(param[1:])+param[0])
        minus_predict = self.link_func(self.X_minus.dot(param[1:])+param[0])
        #loss for positive
        loss = -0.5*np.mean(plus_predict)
        loss += 0.5*np.mean(minus_predict)
        loss -= np.mean((1.0-2*unlabeled_predict)*unlabeled_predict + unlabeled_predict*unlabeled_predict)
        loss += self.reg * np.sum(param*param)
        return loss

    def fit(self, org_X, org_y, cdf_func, n_sample = 1000, use_unlabeled = True,use_simplified = True, rand_seed = 42, reg = 0.0):

        self.cdf_func = cdf_func
        self.reg = reg
        Compara_X, Compara_y = build_dataset(org_X, org_y, n_sample)
        nDim = org_X.shape[1]
        self.X_plus, self.X_minus = transform_dataset_to_CU(Compara_X, Compara_y)
        if(use_unlabeled):
            self.X_unlabeled = np.array(org_X)
        else:
            self.X_unlabeled = np.r_[self.X_plus, self.X_minus]
        if(use_simplified):
            self.link_func = expit
        else:
            self.link_func = self.cdf_func
        x0 = np.zeros(nDim+1)
        params = minimize(score, x0, args = (self,))
        self.param = params["x"]

    #deprecated
    def train_linear(self, X_unlabeled, X_plus, X_minus, reg=0.0, use_simplified = False):
        self.X_unlabeled = X_unlabeled
        self.X_plus = X_plus
        self.X_minus = X_minus
        self.reg = 0.0
        if(use_simplified):
            self.link_func = expit
        else:
            self.link_func = self.cdf_func
        x0 = np.zeros(X_unlabeled.shape[1]+1)
        params = minimize(score, x0, args = (self,))
        self.param = params["x"]


    def predict_val(self, X, lowerS = -10, upperS = 10):
        pred = self.link_func(X.dot(self.param[1:])+self.param[0])
        return np.array([find_quantile_one(self.cdf_func,p,lowerS,upperS) for p in pred])

    def predict_prob(self, X):
        pred = self.link_func(X.dot(self.param[1:])+self.param[0])
        return pred

    def predict_org(self, X):
        return X.dot(self.param[1:]) + self.param[0]





if __name__ == "__main__" :
    mdl = CUmodel()
    mdl.set_KDE_CDF(np.array([1.0,2.0,3.0,4.0,5.0]))
    print(mdl.cdf_func(np.array([1.0,2.0,3.0])))
    print(mdl.cdf_func(2.0))
    print(mdl.find_quantile_one(0.5))

    

    
