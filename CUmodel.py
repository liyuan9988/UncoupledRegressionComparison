import numpy as np
from utils import find_quantile_one
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import logging
logger = logging.getLogger(__name__)

class CULoss(nn.Module):

    def __init__(self, lam):
        self.lam = lam
        super(CULoss, self).__init__()

    def forward(self, unlabeled_input, plus_input, minus_input):
        loss = -self.lam * torch.mean(plus_input)
        loss += (1-self.lam) * torch.mean(minus_input)
        loss -= torch.mean((1-self.lam-unlabeled_input)*2.0*unlabeled_input + unlabeled_input*unlabeled_input)
        return loss

class CUmodelT(nn.Module):
    def __init__(self, nDim, link_func):
        super(CUmodelT, self).__init__()
        self.weight = nn.Linear(nDim, 1)
        self.link_func = link_func

    def forward(self, input):
        pred = self.weight(input)
        if(self.link_func is not None):
            pred = self.link_func(pred)
        return pred

class CUmodel:
    def __init__(self):
        pass

    def fit(self, X_plus, X_minus, cdf_func, X_unlabeled = None, link_func = None, reg = 0.0, lam = 0.5, n_epochs = 3):
        self.cdf_func = cdf_func
        self.reg = reg
        nDim = X_plus.shape[1]
        if(X_unlabeled is not None):
            self.X_unlabeled = X_unlabeled
        else:
            self.X_unlabeled = np.r_[X_plus, X_minus]

        self.model = CUmodelT(nDim, link_func)
        
        optimizer = optim.LBFGS(self.model.parameters())
        criterion = CULoss(lam)
        t_X_plus = torch.tensor(X_plus).float()
        t_X_minus = torch.tensor(X_minus).float()
        t_X_unlabeled = torch.tensor(X_unlabeled).float()
        logger.debug("start learning")
        for epoch in range(n_epochs):
            def closure():
                optimizer.zero_grad()
                plus_input = self.model(t_X_plus)
                minus_input = self.model(t_X_minus)
                unlabel_input = self.model(t_X_unlabeled)
                loss = criterion(unlabel_input, plus_input, minus_input)
                decay = torch.sum(self.model.weight._parameters["weight"]*self.model.weight._parameters["weight"])
                loss += decay * reg
                loss.backward()
                return loss
            optimizer.step(closure)
            logger.debug("Epoch %d, loss %f"%(epoch, closure().item()))
        logger.info("CU fit finished")

    def predict_val(self, X, lowerS = -10, upperS = 10):
        pred = self.model(torch.tensor(X).float()).detach().numpy()[:,0]
        pred = find_quantile_one(self.cdf_func, pred, lowerS, upperS)
        logger.info("CU predict finished")
        return pred
    
    def predict_org(self, X):
        pred =  self.model(torch.tensor(X).float()).detach().numpy()[:,0]
        logger.info("CU predict finished")
        return pred
        
   




if __name__ == "__main__" :
    mdl = CUmodel()
    mdl.set_KDE_CDF(np.array([1.0,2.0,3.0,4.0,5.0]))
    print(mdl.cdf_func(np.array([1.0,2.0,3.0])))
    print(mdl.cdf_func(2.0))
    print(mdl.find_quantile_one(0.5))

    

    
