import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys, rich, torch, time, random
import numpy as np

sys.path.append('./')
from Tasks.Trainer import Trainer
from Data.preprocess import *
from Utils.configs import GAUR
from Utils.logger import get_rich_logger
from Utils.wandb import configure_wandb

def main():
    
    # configuration
    config = GAUR.parse_arguments()
    
    # Seed Fix
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    """Configs Printed and Saved"""
    rich.print(config.__dict__)


    ckpt_dir = os.path.join(config.checkpoint_dir) + "_seed" + str(config.seed)
    config.save(ckpt_dir)
    
    rich.print(f"Training Start")
    main_worker(config.gpus, ckpt_dir, config = config)
    
    
def main_worker(local_rank:int, ckpt_dir, config:object):
    # Set default gpus number
    torch.cuda.set_device(local_rank)

    # Prepare dataset
    data_dir = os.path.join(config.data_dir, config.data)
    if config.data == 'Dalia':
        data = preprocess_dalia(data_dir)
    else:
        data = preprocess_ieee(data_dir, config.band_pass_low, config.band_pass_high, config.band_pass_order, config.resample_fs, config.data)
        data['X'] = np.concatenate([np.expand_dims(data['X'][:, 0,:], axis=1), np.expand_dims(data['X'][:, 1,:], axis=1)], axis=1)
    
    num_subjects = len(np.unique(data['subject']))
    
    # Checkpoint dir
    os.makedirs(ckpt_dir, exist_ok=True)
    logfile = os.path.join(ckpt_dir, 'main.log')
    logger = get_rich_logger(logfile=logfile)
    logger.info(f'Checkpoint directory: {ckpt_dir}')
    
    # Wandb
    if config.enable_wandb:
            configure_wandb(
                name=f'{config.task} : {config.hash}',
                project=f'JH-{config.data}-{config.task}',
                config=config
            )

    # Model Train & Evaluate
    start = time.time()
    
    model = Trainer(
        config = config,
        device=local_rank,
        seed=config.seed,
        ckpt_dir=ckpt_dir,
        num_subjects = num_subjects,
        batch_size = config.batch_size,
        input_dim = data['X'].shape[1]
    )
    
    model.run(data=data, logger=logger)
    model.save_testing_result(config)
    
    end_sec = time.time() - start
    
    if logger is not None:
        end_min = end_sec / 60
        logger.info(f"Total Training Time: {end_min: 2f} minutes")
        logger.handlers.clear()
        

if __name__ == '__main__':
    try:
        main()
        
    except KeyboardInterrupt:
        sys.exit(0)