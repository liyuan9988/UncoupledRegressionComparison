import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import logging
logger = logging.getLogger(__name__)


class TaylorCULoss(nn.Module):

    def __init__(self, lam, taylor_point, dense_at_taylor, cdf_at_taylor):
        self.lam = lam
        self.taylor_point = taylor_point
        self.dense_at_taylor =  dense_at_taylor
        self.cdf_at_taylor =  cdf_at_taylor
        logger.debug((taylor_point, dense_at_taylor, cdf_at_taylor))
        super(TaylorCULoss, self).__init__()

    def forward(self, unlabeled_input, plus_input, minus_input):
        loss = -self.lam * torch.mean(plus_input)/self.dense_at_taylor
        loss += (1-self.lam) * torch.mean(minus_input) /self.dense_at_taylor
        loss -= torch.mean((self.taylor_point + (1-self.lam-self.cdf_at_taylor)/self.dense_at_taylor-unlabeled_input)*2.0*unlabeled_input + unlabeled_input*unlabeled_input)
        return loss

class TaylorCUmodel:
    def __init__(self):
        pass
    
    def fit(self, X_plus, X_minus, taylor_point, dense_at_taylor, cdf_at_taylor,X_unlabeled = None,  reg = 0.0, lam = 0.5, n_epochs = 1):
        nDim = X_plus.shape[1]
        if(X_unlabeled is not None):
            self.X_unlabeled = X_unlabeled
        else:
            self.X_unlabeled = np.r_[X_plus, X_minus]
        self.model = nn.Linear(nDim,1)
        optimizer = optim.LBFGS(self.model.parameters())
        criterion = TaylorCULoss(lam, taylor_point, dense_at_taylor, cdf_at_taylor)
        t_X_plus = torch.tensor(X_plus).float()
        t_X_minus = torch.tensor(X_minus).float()
        t_X_unlabeled = torch.tensor(X_unlabeled).float()
        logger.debug("start learning")
        logger.debug("start learning")
        for epoch in range(n_epochs):
            def closure():
                optimizer.zero_grad()
                plus_input = self.model(t_X_plus)
                minus_input = self.model(t_X_minus)
                unlabel_input = self.model(t_X_unlabeled)
                loss = criterion(unlabel_input, plus_input, minus_input)
                loss.backward()
                return loss
            optimizer.step(closure)
            logger.debug("Epoch %d, loss %f"%(epoch, closure().item()))
        logger.info("TaylorCU fit finished")

    def predict_val(self, X):
        pred =  self.model(torch.tensor(X).float()).detach().numpy()[:,0]
        logger.info("TaylorCU predict finished")
        return pred
        




if __name__ == "__main__" :
    mdl = CUmodel()
    mdl.set_KDE_CDF(np.array([1.0,2.0,3.0,4.0,5.0]))
    print(mdl.cdf_func(np.array([1.0,2.0,3.0])))
    print(mdl.cdf_func(2.0))
    print(mdl.find_quantile_one(0.5))

    

    
