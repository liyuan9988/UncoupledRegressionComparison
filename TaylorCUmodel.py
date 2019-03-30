import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from utils import find_quantile_one, build_dataset, transform_dataset_to_CU

def score(x, mdl):
    return mdl._score(x)

class TaylorCUmodel:
    def __init__(self):
        pass
    
    def _score(self, param):
        unlabeled_predict = self.X_unlabeled.dot(param[1:]) + param[0]
        plus_predict = self.X_plus.dot(param[1:])+param[0]
        minus_predict = self.X_minus.dot(param[1:])+param[0]
        #loss for positive
        loss = -0.5*np.mean(plus_predict)/self.dense_at_taylor
        loss += 0.5*np.mean(minus_predict)/self.dense_at_taylor
        loss -= np.mean((self.taylor_point + ((0.5-self.cdf_at_taylor)/self.dense_at_taylor)-unlabeled_predict)*2*unlabeled_predict + unlabeled_predict*unlabeled_predict)
        #reg term
        loss += self.reg * np.sum(param*param)
        return loss

    
    def fit(self, org_X, org_y, taylor_point, dense_at_taylor, cdf_at_taylor, n_sample = 1000, use_unlabeled = True, rand_seed = 42, reg = 0.0):

        self.taylor_point = taylor_point
        self.dense_at_taylor =  dense_at_taylor
        self.cdf_at_taylor =  cdf_at_taylor
        self.reg = reg
        Compara_X, Compara_y = build_dataset(org_X, org_y, n_sample)
        nDim = org_X.shape[1]
        self.X_plus, self.X_minus = transform_dataset_to_CU(Compara_X, Compara_y)
        if(use_unlabeled):
            self.X_unlabeled = org_X
        else:
            self.X_unlabeled = np.r_[self.X_plus, self.X_minus]
        x0 = np.zeros(nDim+1)
        params = minimize(score, x0, args = (self,))
        self.param = params["x"]

    def predict_val(self, X):
        return X.dot(self.param[1:]) + self.param[0]

    def predict_org(self, X):
        return X.dot(self.param[1:]) + self.param[0]





if __name__ == "__main__" :
    mdl = CUmodel()
    mdl.set_KDE_CDF(np.array([1.0,2.0,3.0,4.0,5.0]))
    print(mdl.cdf_func(np.array([1.0,2.0,3.0])))
    print(mdl.cdf_func(2.0))
    print(mdl.find_quantile_one(0.5))

    

    
