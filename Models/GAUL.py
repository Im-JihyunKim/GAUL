import torch
import torch.nn as nn
from Utils.loss import weighted_loss
from Utils.solver_utils import *

class GAUL_Net(nn.Module):
    def __init__(self, device, config, num_sources, G, R1, R2):
        super(GAUL_Net, self).__init__()
        self.device = device
        self.config = config
        self.n_sources = num_sources

        self.loss = weighted_loss(config.reduction)

        self.G, self.R1, self.R2 = G, R1, R2

        self.yt_mean, self.yt_std = None, None
        self.max_sample, self.min_sample = None, None
        self.ys_mean = []

    def forward(self, Xs, Xt):
        # Feature Extractor
        sx = extract_source_features(self.n_sources, self.config, self.G, Xs)
        tx = extract_target_features(self.config, self.G, Xt)

        # Predictors
        ys_1, yt_1 = predictor1(self.n_sources, self.R1, sx, tx)
        ys_2, yt_2 = predictor2(self.n_sources, self.R2, sx, tx)

        # Dual-level Weighting
        with torch.no_grad():
            yt_mean, yt_std = torch.mean((yt_1+yt_2)/2, dim=0), torch.std((yt_1+yt_2)/2, dim=0)        # target pred y -> mean & std
            self.yt_mean = yt_mean if self.yt_mean is None else ema_update(self.config.ema_beta, self.yt_mean, yt_mean) # ema update
            self.yt_std = yt_std if self.yt_std is None else ema_update(self.config.ema_beta, self.yt_std, yt_std)

            ys_pred = [(ys_1[i]+ys_2[i])/2 for i in range(self.n_sources)]  # source pred
            ys_mean = self.get_source_centroid(ys_pred)                     # source y_hat > mean

            domain_weight, sample_weight = self.get_weight(ys_pred, ys_mean)

        return ys_1, yt_1, ys_2, yt_2, domain_weight, sample_weight
    
    def train_step1(self, Xs, Xt, ys):
        self.train()
        ys_1, _, ys_2, _, domain_weight, sample_weight = self.forward(Xs, Xt)

        loss_R1 = self.loss.pred_loss(ys_1, ys, domain_weight, sample_weight)
        loss_R2 = self.loss.pred_loss(ys_2, ys, domain_weight, sample_weight)

        loss_pred = loss_R1 + loss_R2
        loss_pred.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)

        self.opt_G.step()
        self.opt_R1.step()
        self.opt_R2.step()
        self.reset_grad()

        return loss_pred
    
    def train_step2(self, Xs, Xt, ys):
        self.train()
        ys_1, yt_1, ys_2, yt_2, domain_weight, sample_weight = self.forward(Xs, Xt)

        loss_R1 = self.loss.pred_loss(ys_1, ys, domain_weight, sample_weight)
        loss_R2 = self.loss.pred_loss(ys_2, ys, domain_weight, sample_weight)

        loss_pred = loss_R1 + loss_R2
        loss_disc = self.loss.disc_loss(ys_1, ys_2, yt_1, yt_2, domain_weight, sample_weight)

        loss = loss_pred - self.config.mu*loss_disc
        loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)

        self.opt_R1.step()
        self.opt_R2.step()
        self.reset_grad()

    def train_step3(self, Xs, Xt):
        self.train()
        ys_1, yt_1, ys_2, yt_2, domain_weight, sample_weight = self.forward(Xs, Xt)

        loss_disc = self.loss.disc_loss(ys_1, ys_2, yt_1, yt_2, domain_weight, sample_weight)
        loss = self.config.mu*loss_disc 
        loss.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(self.parameters(), self.config.clip)

        self.opt_G.step()
        self.reset_grad()

        return loss_disc
    
    def predict(self, Xt):
        Xt_feat = extract_target_features(self.config, self.G, Xt)
        target_pred = predictor1(self.n_sources, self.R1, None, Xt_feat)[1]
        return target_pred

    def get_weight(self, ys:list, ys_mean:list):
        yt_dist = torch.distributions.Normal(self.yt_mean, self.yt_std)
        w_sample = torch.stack([torch.exp(yt_dist.log_prob(ys[i])) for i in range(self.n_sources)])  
        w_alpha = torch.stack([torch.exp(yt_dist.log_prob(ys_mean[i])) for i in range(self.n_sources)]).squeeze() 
        
        # Normalization
        self.max_sample = w_sample.max() if self.max_sample is None else ema_update(self.config.ema_beta, self.max_sample, w_sample.max())
        self.min_sample = w_sample.min() if self.min_sample is None else ema_update(self.config.ema_beta, self.min_sample, w_sample.min())
        domain_weight = (w_alpha+1e-6)/(w_alpha+1e-6).max()
        domain_weight.div_(torch.norm(domain_weight, p=1))
        sample_weight = torch.exp(((w_sample-self.min_sample) / (self.max_sample-self.min_sample+1e-6)).clip(-torch.inf, 88))

        return domain_weight, sample_weight
    
    def get_source_centroid(self, ys_pred: list):
        for i in range(self.n_sources):
            ys_mean = ys_pred[i].clone()
            if len(self.ys_mean) != self.n_sources:
                self.ys_mean.append(torch.mean(ys_mean, dim=0))
            else:
                self.ys_mean[i] = (1-self.config.ema_beta)*torch.mean(ys_mean, dim=0) + self.config.ema_beta*self.ys_mean[i].data
        return self.ys_mean

    def set_optimizers(self):
        if self.config.optimizer == "adam":
            self.opt_G = torch.optim.Adam([{'params': self.G.parameters()}], lr=self.config.lr, 
                                        betas=(0.9, 0.99), weight_decay=self.config.weight_decay)
            self.opt_R1 = torch.optim.Adam([{'params': self.R1.parameters()}], lr=self.config.lr, 
                                        betas=(0.9, 0.99), weight_decay=self.config.weight_decay)
            self.opt_R2 = torch.optim.Adam([{'params': self.R2.parameters()}], lr=self.config.lr, 
                                        betas=(0.9, 0.99), weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "sgd":
            self.opt_G = torch.optim.SGD([{'params': self.G.parameters()}], lr=self.config.lr, 
                                        momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            self.opt_R1 = torch.optim.SGD([{'params': self.R1.parameters()}], lr=self.config.lr, 
                                        momentum=self.config.momentum, weight_decay=self.config.weight_decay)
            self.opt_R2 = torch.optim.SGD([{'params': self.R2.parameters()}], lr=self.config.lr, 
                                        momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            raise NotImplementedError
        
    def set_schedulers(self):
        if self.config.lr_scheduler == "step":
            step_size = int(self.config.epochs*(1/3))
            G_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_G,
                                                    step_size=step_size, gamma=0.5)
            R1_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_R1,
                                                    step_size=step_size, gamma=0.5)
            R2_scheduler = torch.optim.lr_scheduler.StepLR(self.opt_R2,
                                                    step_size=step_size, gamma=0.5)
            return G_scheduler, R1_scheduler, R2_scheduler
        else:
            raise NotImplementedError
        
    def reset_grad(self):
        self.opt_G.zero_grad()
        self.opt_R1.zero_grad()
        self.opt_R2.zero_grad()
