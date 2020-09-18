import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import logging
logger = logging.getLogger(__name__)

def score(x, mdl):
    return mdl._score(x)

class OptimizedTaylorCUmodel:
    def __init__(self):
        pass
    
    def _score(self, param):
        unlabeled_predict = self.X_unlabeled.dot(param[1:]) + param[0]
        plus_predict = self.X_plus.dot(param[1:])+param[0]
        minus_predict = self.X_minus.dot(param[1:])+param[0]
        #loss for positive
        loss = -2*np.mean(unlabeled_predict) * self.weight[0]
        loss -= np.mean(plus_predict) * self.weight[1]
        loss -= np.mean(minus_predict)* self.weight[2]
        loss += np.mean(unlabeled_predict*unlabeled_predict)
        #reg term
        loss += self.reg * np.sum(param*param)
        return loss

    def derive_optimal_const(self, min_y, max_y, n_split = 1000):
        try_y_array = np.linspace(min_y, max_y, n_split)
        dense_array = self.dense_func(try_y_array)
        cdf_array = self.cdf_func(try_y_array)
        X = np.c_[cdf_array, 1.0-cdf_array]
        mdl = LinearRegression(fit_intercept=False)
        mdl.fit(X,try_y_array,sample_weight=dense_array)
        tmp = np.sum(mdl.coef_)
        logger.debug(tmp)
        n_unlabel = self.X_unlabeled.shape[0]
        n_rank = self.X_plus.shape[0]
        lam = -(n_unlabel*tmp)/(4*n_rank + n_unlabel*2)
        self.weight = np.array([-lam, mdl.coef_[0]+lam, mdl.coef_[1]+lam])
        logger.debug("original coef:"+str(mdl.coef_))
        logger.debug("optimized weight:"+str(self.weight))

    def fit(self, X_plus, X_minus, cdf_func, dense_func, min_y, max_y,  X_unlabeled = None, 
    reg = 0.0, optimal_weight = None):
        self.dense_func =  dense_func
        self.cdf_func =  cdf_func
        self.reg = reg
        self.X_plus = X_plus
        self.X_minus = X_minus
        nDim = X_plus.shape[1]
        if(X_unlabeled is not None):
            self.X_unlabeled = np.array(X_unlabeled)
        else:
            self.X_unlabeled = np.r_[self.X_plus, self.X_minus]
        if(optimal_weight is None):
            self.derive_optimal_const(min_y, max_y)
        else:
            self.weight = optimal_weight
        logger.debug(self.weight) 
        x0 = np.zeros(nDim+1)
        logger.debug("start training")
        params = minimize(score, x0, args = (self,))
        self.param = params["x"]
        logger.debug("end training")

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

    

    
