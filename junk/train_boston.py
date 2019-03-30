import numpy as np
from CUmodel import CUmodel
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
np.random.seed(42)

def generate_data_for_CU(N_plus):
    boston = load_boston()
    X = np.array(boston.data, copy=True)
    y = np.array(boston.target, copy=True)
    X_unlabeled, X_test, Y_unlabeled, Y_test = train_test_split(X, y, test_size=0.33)
    nData, nDim = X_unlabeled.shape
    X_plus = np.empty((N_plus,nDim))
    X_minus = np.empty((N_plus,nDim))
    for i in range(N_plus):
        tmp1 = np.random.choice(nData)
        tmp2 = np.random.choice(nData)
        if(Y_unlabeled[tmp2] > Y_unlabeled[tmp1]):
            tmp1, tmp2 = tmp2, tmp1
        X_plus[i] = X_unlabeled[tmp1]
        X_minus[i] = X_unlabeled[tmp2]
    return X_unlabeled, Y_unlabeled, X_plus, X_minus, X_test, Y_test


def test_CUmodel_and_LR(n_iter, N_plus):
    lr_res = []
    cu_res = []
    org_cu_res = []
    for counter in range(n_iter):
        X_unlabeled, Y_unlabeled, X_plus, X_minus, X_test, Y_test= generate_data_for_CU(N_plus)
        #fit linear
        lr_model = LinearRegression()
        lr_model.fit(X_unlabeled, Y_unlabeled)
        est_err = Y_test - lr_model.predict(X_test)
        lr_res.append(np.mean(est_err*est_err))
        #fit CUmodel
        cu_model = CUmodel()
        cu_model.set_KDE_CDF(Y_unlabeled, sigma = 5.0)
        cu_model.train_linear(X_unlabeled, X_plus, X_minus,use_simplified=True)
        est_err = Y_test - cu_model.predict_val(X_test,0.0,100)
        cu_res.append(np.mean((est_err*est_err)))
        est_err = Y_test - cu_model.predict_org(X_test)
        org_cu_res.append(np.mean((est_err*est_err)))
    return np.array(lr_res), np.array(cu_res), np.array(org_cu_res)