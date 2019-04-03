import numpy as np
from scipy.optimize import minimize
from utils import find_quantile_one
import logging
logger = logging.getLogger(__name__)

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
        loss -= np.mean(unlabeled_predict - unlabeled_predict*unlabeled_predict)
        loss += self.reg * np.sum(param*param)
        return loss


    def fit(self, X_plus, X_minus, cdf_func, link_func, X_unlabeled = None, reg = 0.0):

        self.cdf_func = cdf_func
        self.reg = reg
        nDim = X_plus.shape[1]
        self.X_plus = X_plus
        self.X_minus = X_minus
        if(X_unlabeled is not None):
            self.X_unlabeled = np.array(X_unlabeled)
        else:
            self.X_unlabeled = np.r_[self.X_plus, self.X_minus]
        self.link_func = link_func
        x0 = np.zeros(nDim+1)
        logger.debug("start training")
        params = minimize(score, x0, args = (self,))
        logger.debug(params)
        self.param = params["x"]
        logger.debug("end training")

    def predict_val(self, X, lowerS = -10, upperS = 10):
        pred = self.link_func(X.dot(self.param[1:])+self.param[0])
        #return np.array([find_quantile_one(self.cdf_func,p,lowerS,upperS) for p in pred])
        return find_quantile_one(self.cdf_func,pred,lowerS,upperS)

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

    

    
