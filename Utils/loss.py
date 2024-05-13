import torch
import torch.nn as nn
import torch.nn.functional as F

class weighted_loss(nn.Module):
    def __init__(self, reduction):
        super(weighted_loss, self).__init__()
        self.reduction = reduction       # reduction - summation or mean

    def pred_loss(self, pred, target, d, s):  # d: domain weight, s: sample weight
        loss = get_loss(loss="logcosh", reduction="none")
        w_loss = torch.stack([torch.mean(d[i]*(s[i]*loss(pred[i], target[i]))) for i in range(len(pred))])
        return torch.sum(w_loss) if self.reduction=="sum" else torch.mean(w_loss)
    
    def disc_loss(self, ys_1, ys_2, yt_1, yt_2, d, s): # d: domain weight, s: sample weight
        loss = get_loss(loss="mse", reduction="mean")
        source_disc = torch.stack([torch.mean(d[i]*(s[i]*loss(ys_2[i], ys_1[i]))) for i in range(len(ys_1))])
        w_disc = torch.abs(torch.sum(source_disc) - loss(yt_2, yt_1))
        return w_disc

class get_loss(nn.Module):
    def __init__(self, loss:str="mse", reduction:str="none"):
        super(get_loss, self).__init__()
        self.loss = loss
        self.reduction = reduction

    def forward(self, pred:torch.Tensor, target:torch.Tensor):
        if self.loss == "mse":
            loss = F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss == 'logcosh':
            loss = self.logcosh(pred, target, reduction=self.reduction)
        else:
            raise NotImplementedError
        return loss
    
    @staticmethod
    def logcosh(pred, target, reduction):
        x = pred - target
        log_cosh = (x + torch.nn.Softplus()(-2*x) - torch.log(torch.tensor(2.)))
        if reduction == "mean":
            log_cosh = log_cosh.mean()
        elif reduction == "sum":
            log_cosh = log_cosh.sum()
        return log_cosh