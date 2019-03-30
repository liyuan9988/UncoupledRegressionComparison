import numpy as np
from utils import build_dataset,find_quantile_one
from sklearn.svm import LinearSVC


class SVMRank:

    def __init__(self):
        pass

    def fit(self, org_X, org_y, cdf_func, n_sample, random_seed = 42, use_unlabeled = False):
        self.cdf_func = cdf_func
        nDim = org_X.shape[1]
        Compara_X, Compara_y = build_dataset(org_X, org_y, n_sample)
        self.mdl = LinearSVC()
        if(use_unlabeled):
            self.X_unlabeled = org_X
        else:
            self.X_unlabeled = np.r_[Compara_X[:,:nDim], Compara_X[:,nDim:]]
        diff_X = Compara_X[:,:nDim]-Compara_X[:,nDim:]
        self.mdl.fit(diff_X,Compara_y)
    
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

