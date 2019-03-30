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

class CUmodel:
    def __init__(self):
        pass

    def fit(self, X_plus, X_minus, cdf_func, X_unlabeled = None, link_func = "linear", reg = 0.0, lam = 0.5, lr=0.001, momentum=0.9, unlabel_batch = 100, n_epochs = 1):
        self.cdf_func = cdf_func
        self.reg = reg
        nDim = X_plus.shape[1]
        if(X_unlabeled is not None):
            self.X_unlabeled = X_unlabeled
        else:
            self.X_unlabeled = np.r_[X_plus, X_minus]

        if(link_func == "linear"):
            self.model = nn.Linear(nDim,1)
        elif(link_func == "sigmoid"):
            self.model = nn.Sequential(nn.Linear(nDim,1), nn.Sigmoid())
        else:
            raise ValueError("invalid link func :%s"%link_func)

        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay = reg, )
        criterion = CULoss(lam)
        t_X_plus = torch.tensor(X_plus).float()
        t_X_minus = torch.tensor(X_minus).float()
        t_X_unlabeled = torch.tensor(X_unlabeled).float()
        unlabel_loader = DataLoader(TensorDataset(t_X_unlabeled), unlabel_batch)
        logger.debug("start learning")
        for epoch in range(n_epochs):
            for (unlabel_input,) in unlabel_loader:
                optimizer.zero_grad()
                plus_input = self.model(t_X_plus)
                minus_input = self.model(t_X_minus)
                unlabel_input = self.model(unlabel_input)
                loss = criterion(unlabel_input, plus_input, minus_input)
                loss.backward()
                optimizer.step()

            logger.debug("Epoch %d, loss %f"%(epoch, loss.item()))
        logger.info("CU fit finished")

    def predict_val(self, X, lowerS = -10, upperS = 10):
        pred = self.model(torch.tensor(X).float()).detach().numpy()[:,0]
        pred = find_quantile_one(self.cdf_func, pred, lowerS, upperS)
        logger.info("CU predict finished")
        return pred
   




if __name__ == "__main__" :
    mdl = CUmodel()
    mdl.set_KDE_CDF(np.array([1.0,2.0,3.0,4.0,5.0]))
    print(mdl.cdf_func(np.array([1.0,2.0,3.0])))
    print(mdl.cdf_func(2.0))
    print(mdl.find_quantile_one(0.5))

    

    
