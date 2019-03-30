import numpy as np
from utils import find_quantile_one
from sklearn.svm import LinearSVC
import logging
logger = logging.getLogger(__name__)


class SVMRank:

    def __init__(self):
        pass

    def fit(self, Compara_X, Compara_y, cdf_func, unlabel_X = None):
        nDim = Compara_X.shape[1] // 2
        self.cdf_func = cdf_func
        self.mdl = LinearSVC()
        if(unlabel_X is not None):
            self.X_unlabeled = unlabel_X
        else:
            self.X_unlabeled = np.r_[Compara_X[:,:nDim], Compara_X[:,nDim:]]
        diff_X = Compara_X[:,:nDim]-Compara_X[:,nDim:]
        logger.info("SVMRank: start fitting")
        self.mdl.fit(diff_X,Compara_y)
        logger.info("SVMRank: end fitting")
    
    def predict_score(self,X):
        return self.mdl.predict(X)
    
    def predict_unlabeled(self,lowerS = -10.0, upperS = 10.0):
        return self.predict_core(self.X_unlabeled,lowerS, upperS)

    def predict_val(self,X,lowerS = -10.0, upperS = 10.0):
        nData = X.shape[0]
        X_ref = np.r_[X, self.X_unlabeled]
        res = self.predict_core(X_ref,lowerS, upperS)
        return res[:nData]

    def predict_core(self,X ,lowerS = -10.0, upperS = 10.0):
        score = self.predict_score(X)
        rank = np.argsort(score)
        prob = (rank+1.0)/(X.shape[0]+1.0)
        return np.array([find_quantile_one(self.cdf_func,p,lowerS, upperS) for p in prob])

