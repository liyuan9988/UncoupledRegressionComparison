import numpy as np
from CUmodel import CUmodel
from sklearn.linear_model import LinearRegression
np.random.seed(42)

def generate_data_for_CU(nDim, N_unlabeled, N_plus, N_test, theta):
    X_unlabeled = np.random.normal(0,1,(N_unlabeled,nDim))
    Y_unlabeled = X_unlabeled.dot(theta) + np.random.normal(0,0.1,N_unlabeled)
    X_test = np.random.normal(0,1,(N_test,nDim))
    Y_test = X_test.dot(theta) + np.random.normal(0,0.1,N_test)
    X_plus = np.empty((N_plus,nDim))
    X_minus = np.empty((N_plus,nDim))
    for i in range(N_plus):
        tmp = np.random.normal(0,1,(2,nDim))
        y0 = tmp[0].dot(theta) + np.random.normal(0,0.1)
        y1 = tmp[1].dot(theta) + np.random.normal(0,0.1)
        if(y0>y1):
            X_plus[i] = tmp[0]
            X_minus[i] = tmp[1]
        else:
            X_plus[i] = tmp[1]
            X_minus[i] = tmp[0]
    return X_unlabeled, Y_unlabeled, X_plus, X_minus, X_test, Y_test


def test_CUmodel_and_LR(nDim, n_iter, N_unlabeled, N_plus):
    lr_res = []
    cu_res = []
    org_cu_res = []
    for counter in range(n_iter):
        theta = np.random.normal(0,1,nDim)
        theta /= np.linalg.norm(theta)
        X_unlabeled, Y_unlabeled, X_plus, X_minus, X_test, Y_test= generate_data_for_CU(nDim, N_unlabeled, N_plus, 100, theta)
        #fit linear
        lr_model = LinearRegression()
        lr_model.fit(X_unlabeled, Y_unlabeled)
        est_err = Y_test - lr_model.predict(X_test)
        lr_res.append(np.mean(est_err*est_err))
        #fit CUmodel
        cu_model = CUmodel()
        cu_model.set_normal_CDF(0.0, np.sqrt(1.0 + 0.01))
        cu_model.train_linear(X_unlabeled, X_plus, X_minus)
        est_err = Y_test - cu_model.predict_val(X_test)
        cu_res.append(np.mean((est_err*est_err)))
        est_err = Y_test - cu_model.predict_org(X_test)
        org_cu_res.append(np.mean((est_err*est_err)))
    return np.array(lr_res), np.array(cu_res), np.array(org_cu_res)