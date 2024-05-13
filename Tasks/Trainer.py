import os, rich, collections
import numpy as np
import pandas as pd
import torch 
from torch.utils.tensorboard import SummaryWriter
from rich.progress import Progress

from Models.GAUR import GAURNet
from Models.Networks import feature_extractor, predictor1, predictor2
from Dataloaders import split_source_target, batch_loader
from Utils.logger import make_epoch_description
from Tasks.Base import Task

import warnings
warnings.filterwarnings(action='ignore')

class Trainer(Task):
    def __init__(self, config, device, seed, ckpt_dir, num_subjects,
                **kwargs):
        super(Trainer, self).__init__()
        
        self.config = config
        self.device = device
        self.seed = seed
        self.ckpt_dir = ckpt_dir
        self.num_subjects = num_subjects
        
        self.epochs = config.epochs
        self.epochs_G = config.epochs_G
        self.epochs_pred = config.epochs_pred
        self.epochs_disc = config.epochs_disc

        self.__dict__.update(kwargs)

    def prepare(self, data, test_idx):
        writer = SummaryWriter(os.path.join(self.ckpt_dir, str(test_idx+1)))
        Xs, ys, Xt, yt = split_source_target(data, test_idx=test_idx, device=self.device)
        G = feature_extractor(self.input_dim, self.config.bidirectional, self.config.num_layers)
        R1, R2 = predictor1(self.config.bidirectional), predictor2(self.config.bidirectional)

        return Xs, ys, Xt, yt, G, R1, R2, writer

    def run(self, data, logger):
        alphas = {}
        for test_idx in range(self.num_subjects):
            torch.cuda.empty_cache()
            os.makedirs(os.path.join(self.ckpt_dir, str(test_idx+1)), exist_ok=True)

            Xs, ys, Xt, yt, G, R1, R2, writer = self.prepare(data, test_idx)
            G, R1, R2 = G.to(self.device), R1.to(self.device), R2.to(self.device)
            self.model = GAURNet(self.device, self.config, self.num_subjects-1, G, R1, R2).to(self.device)

            self.model.set_optimizers()  # set optimizer
            if self.config.lr_scheduler is not None:
                G_scheduler, R1_scheduler, R2_scheduler = self.model.set_schedulers()
            
            for epoch in range(1, self.epochs+1):
                train_results, domain_weight = self.train(Xs, ys, Xt, yt)

                # lr scheduler
                if self.config.lr_scheduler is not None:
                    G_scheduler.step()
                    R1_scheduler.step()
                    R2_scheduler.step()
                    rich.print('lr', G_scheduler.get_last_lr()[0])
                
                self.epoch_history = collections.defaultdict(dict)
                for k, v in  train_results.items():
                    self.epoch_history[k]['DA'] = v
                    
                # Write TensorBoard Summary
                for k, v in self.epoch_history.items():
                    for k_, v_ in v.items():
                        writer.add_scalar(f'{k}_{k_}', v_, global_step=epoch)

                # Logging
                log = make_epoch_description(
                    history=self.epoch_history,
                    current=epoch,
                    total=self.epochs,
                    best=epoch
                )
                log += f" Subject_index {test_idx+1} / Total_Subject_index {self.num_subjects} "
                logger.info(log)
                    
            # Save checkpoint when training ends
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, str(test_idx+1), 'best_adaptation.pth'))    

            # test
            self.model.eval()
            mae = torch.sum(torch.abs(yt.squeeze_() - self.model.predict(Xt).squeeze_())).item()/yt.shape[0]
            alphas[test_idx] = domain_weight
            rich.print('MAE:', mae, 'Alpha', alphas[test_idx])
            
            test_results = self.test(test_idx, Xt, yt)
            self.save_testing_result_per_subject(result=test_results, path=os.path.join(self.ckpt_dir, str(test_idx+1)), config=self.config)
    
        # Save results (alpha)
        alpha_df = pd.DataFrame(alphas)
        alpha_df.to_csv(os.path.join(self.ckpt_dir, 'Results_alpha.csv'))

    
    def train(self, Xs, ys, Xt, yt):
        self.model.train()
        trainloader = batch_loader(Xs, ys, batch_size=self.batch_size, seed=self.seed)
        
        max_input_size = max([Xs[i].size(0) for i in range(len(Xs))])
        num_steps = int(max_input_size/self.batch_size)
        result = {'train_loss': torch.zeros(num_steps), 'disc': torch.zeros(num_steps),
                  'R1_S': torch.zeros(num_steps), 'R2_S': torch.zeros(num_steps),
                  'R1_T' : torch.zeros(num_steps), 'R2_T' : torch.zeros(num_steps),
                  'real_yt': torch.zeros(num_steps),'test_mae': torch.zeros(num_steps)}

        with Progress(transient=True, auto_refresh=False) as pg:
            task = pg.add_task(f"[bold yellow] Training...", total=num_steps)

            for i, (xs_batch, ys_batch) in enumerate(trainloader):
                ridx = np.random.choice(Xt.shape[0], self.batch_size)
                xt = Xt[ridx, :, :]
                yt_batch = yt[ridx]

                for e in range(self.epochs_pred):
                    loss_pred = self.model.train_step1(xs_batch, xt, ys_batch)

                for e in range(self.epochs_disc):
                    self.model.train_step2(xs_batch, xt, ys_batch)

                for e in range(self.epochs_G):
                    loss_disc = self.model.train_step3(xs_batch, xt)

                y_spred, y_tpred, y_sdisc, y_tdisc, domain_weight, _ = self.model.forward(xs_batch, xt)
                test_mae = torch.sum(torch.abs(yt.squeeze_() - self.model.predict(Xt).squeeze_())).item()/yt.shape[0]
                
                result['train_loss'][i] = loss_pred.item()
                result['disc'][i] = loss_disc.item()
            
                result['R1_S'][i] = torch.mean(torch.stack(y_spred)).item()
                result['R2_S'][i] = torch.mean(torch.stack(y_sdisc)).item()
                result['R1_T'][i] = torch.mean(y_tpred).item()
                result['R2_T'][i] = torch.mean(y_tdisc).item()
                result['real_yt'][i] = torch.mean(yt_batch).item()
                result['test_mae'][i] = test_mae

                desc = f"[bold grey] [{i}/{num_steps}]: "
                for k, v in result.items():
                    desc += f"{k} : {v[:i+1].mean():.4f} |"
                pg.update(task, advance=1., description=desc)
                pg.refresh()
            
            rich.print(f'alpha = {domain_weight.cpu().detach().numpy().round(3)}')
            adaptation_result = {k: v.mean().item() for k, v in result.items()}
            
        return adaptation_result, domain_weight.cpu().detach().numpy()
   
    @torch.no_grad()
    def test(self, test_idx, Xt, yt):
        self.model.eval()
        result = {
            'subjects': np.ndarray(shape=yt.size()),
            'real_y': np.ndarray(shape=yt.size()),
            'pred_y': np.ndarray(shape=yt.size())
        }
        
        for i in range(yt.size(0)):
            result['subjects'][i] = int(test_idx+1)
            
        result['real_y'] = yt.squeeze_().cpu().numpy()
        result['pred_y'] = self.model.predict(Xt).squeeze_().cpu().detach().numpy()
            
        test_result = {k: np.concatenate(v.reshape(-1, 1), axis=0) for k, v in result.items()}
        
        return test_result