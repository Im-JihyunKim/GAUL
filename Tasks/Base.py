import os, glob
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Malgun Gothic')
import seaborn as sns
sns.set_context("talk")
sns.set_style("white")
sns.set_palette("Pastel1")

from Utils.metrics import return_result

class Task(object):
    def __init__(self):
        self.checkpoint_dir = None

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_model_from_checkpoint(self):
        raise NotImplementedError

    def load_history_from_checkpoint(self):
        raise NotImplementedError
    
    def save_testing_result_per_subject(self, result, path, config):
        result = pd.DataFrame(result)
        
        true = result['real_y']
        pred = result['pred_y']
        
        vmin = np.min(np.concatenate([pred, true])) * 0.95
        vmax = np.max(np.concatenate([pred, true])) * 1.05
        
        plt.figure(figsize=(10, 10))
        p = return_result(true, pred)
        v = list(p.values())
        perf_legend = f'corr : {v[0]:.2f} \nmape : {v[1]:.2f} \nrmse : {v[2]:.2f} \nr2 : {v[3]:.2f} \nmae : {v[4]:.2f}'
        plt.scatter(true, pred, label=perf_legend)
        plt.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='--')
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.legend(loc='upper left')
        plt.title(f"{int(np.unique(result['subjects']).item())}_{config.task}")
        save_path = os.path.join(path, f"{int(np.unique(result['subjects']).item())}_{config.task}.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=350)
        plt.close('all')
        
        p['subject'] = int(np.unique(result['subjects']).item())
        performance = pd.DataFrame(p, index=[0])
        performance.to_csv(os.path.join(path, 'performance.csv'), index=False)
        result.to_csv(os.path.join(path, 'info.csv'), index=False)

    def save_testing_result(self, config):
        outputs_path = glob.glob(os.path.join(self.ckpt_dir, "**", f"info.csv"))
        outputs = pd.concat([pd.read_csv(path) for path in outputs_path])
        summary = outputs.groupby('subjects')[['real_y', 'pred_y']].agg(['mean', 'std'])
        summary.to_csv(os.path.join(self.ckpt_dir, f"{config.task}_result.csv"))
        
        performance_path = glob.glob(os.path.join(self.ckpt_dir, "**", f"performance.csv"))
        performance = pd.concat([pd.read_csv(path) for path in performance_path])
        performance.sort_values(by='subject').to_csv(os.path.join(self.ckpt_dir, f"{config.task}_performance.csv"), index=False)
        
        true = summary[('real_y', 'mean')]
        pred = summary[('pred_y', 'mean')]
        
        vmin = np.min(np.concatenate([pred, true])) * 0.95
        vmax = np.max(np.concatenate([pred, true])) * 1.05
        
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = sns.color_palette("tab20", n_colors=len(outputs_path))
        
        p = dict(performance.mean())
        v = list(p.values())
        perf_legend = f'corr : {v[0]:.2f} \nmape : {v[1]:.2f} \nrmse : {v[2]:.2f} \nr2 : {v[3]:.2f} \nmae : {v[4]:.2f}'
        ax.plot([vmin, vmax], [vmin, vmax], color='grey', linestyle='dashed')
        
        for idx, subject in enumerate(np.unique(outputs['subjects'])):
            ax.scatter(summary.reset_index().iloc[idx][('real_y', 'mean')],
                        summary.reset_index().iloc[idx][('pred_y', 'mean')],
                        color = colors[idx], label=int(subject)
                        )
            
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(perf_legend, fontsize=12)
        
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
        ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
        
        save_path = os.path.join(self.ckpt_dir, f"{config.task}_results.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=350)
        plt.close('all')