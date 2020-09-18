import numpy as np
from scipy.optimize import minimize
from utils import find_quantile_one
import logging
logger = logging.getLogger(__name__)


class LinearShuffle:
    def __init__(self):
        pass

    def _score(self, param):
        unlabeled_predict = self.link_func(self.X_unlabeled.dot(param[1:]) + param[0])
        plus_predict = self.link_func(self.X_plus.dot(param[1:])+param[0])
        minus_predict = self.link_func(self.X_minus.dot(param[1:])+param[0])
        # loss for positive
        loss = -0.5*np.mean(plus_predict)
        loss += 0.5*np.mean(minus_predict)
        loss -= np.mean(unlabeled_predict - unlabeled_predict*unlabeled_predict)
        loss += self.reg * np.sum(param*param)
        return loss

    def fit(self, X_unlabeled, Y_shuffled, rep=100):
        nDim = X_unlabeled.shape[1]
        self.w = np.random.uniform(-1.0,1.0,nDim)
        logger.debug("start training")
        sorted_Y = np.sort(Y_shuffled)
        sorted_Y_inv = np.sort(Y_shuffled)[::-1]
        logger.debug("hoge")
        for i in range(rep):
            pred = X_unlabeled.dot(self.w)
            permutation = np.argsort(pred)
            X_tmp = X_unlabeled[permutation]
            if np.abs(X_tmp.T.dot(sorted_Y)) > np.abs(X_tmp.T.dot(sorted_Y_inv)):
                true_Y = np.copy(sorted_Y)
            else:
                true_Y = np.copy(sorted_Y_inv)
            self.w = np.linalg.solve(X_tmp.T.dot(X_tmp), X_tmp.T.dot(true_Y))
            
        logger.debug(self.w)
        logger.debug("end training")

    def predict_val(self, X, lowerS=-10, upperS=10):
        pred = X.dot(self.w)
        return pred

    def predict_prob(self, X):
        pred = X.dot(self.w)
        return pred

    def predict_org(self, X):
        pred = X.dot(self.w)
        return pred


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    mdl = LinearShuffle()
    X = np.random.random(size=(100, 5))
    w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    Y = X.dot(w)
    np.random.shuffle(Y)
    mdl.fit(X, Y)
    print(mdl.w)
