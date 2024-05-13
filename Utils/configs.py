import os, json, copy, datetime, argparse

class ConfigBase(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):
        
        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()
            
        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)
            
        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        parents = [
            cls.data_parser(),
            cls.model_parser(),
            cls.train_parser(),
            cls.logging_parser(),
            cls.task_specific_parser()
        ]
        
        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args
        
        config = cls()
        parser.parse_args(namespace=config)
        
        return config
    
    @classmethod
    def form_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)
    
    def save(self, ckpt_dir):
        path = os.path.join(ckpt_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['task'] = self.task
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)
            
    @property
    def task(self):
        raise NotImplementedError
    
    @property
    def checkpoint_dir(self) -> str:
        ckpt = os.path.join(
            self.checkpoint_root,
            self.task,
            self.data,
            self.hash
        )
        return ckpt
    
    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        raise NotImplementedError
    
    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg
            
    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data-dir', type=str, default='./Data')
        parser.add_argument('--data', type=str, default='IEEE_TEST', choices=('Dalia', 'IEEE_TRAIN', 'IEEE_TEST'))

        parser.add_argument('--band-pass-low', type=float, default=0.4)
        parser.add_argument('--band-pass-high', type=float, default=4.)
        parser.add_argument('--band-pass-order', type=int, default=2)
        parser.add_argument('--resample-fs', type=float, default=25)
        
        return parser
    
    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("PPG-HR Prediction Model", add_help=False)
        parser.add_argument('--bidirectional', type=bool, default=True)
        parser.add_argument('--num-layers', type=int, default=2)
        return parser
    
    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size.')
        parser.add_argument("--epochs", default=300, type=int, help="epochs for adaptation")

        parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate to start from.')
        parser.add_argument("--lr-scheduler", default=None, choices=(None, "step"))

        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--weight-decay", default=0, type=float, help='l2 weight decay')
        parser.add_argument("--clip", default=1, type=int, help="Norm of Gradient Clipping")
    
        parser.add_argument('--optimizer', type=str, default='adam', choices=('sgd', 'adam'), help='Optimization algorithm.')
        parser.add_argument('--patience', type=int, default=0)
        parser.add_argument('--gpus', type=int, nargs='+', default=0, help='')
        return parser
    
    @staticmethod
    def logging_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing logging-related arguments."""
        parser = argparse.ArgumentParser("Logging", add_help=False)
        parser.add_argument('--checkpoint-root', type=str, default='./Results/', help='Top-level directory of checkpoints.')
        parser.add_argument('--enable-wandb', action='store_true', help='Use Weights & Biases plugin.')

        return parser

class GAUR(ConfigBase):
    def __init__(self, args=None, **kwargs):
        super(GAUR, self).__init__(args, **kwargs)
        
    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('', add_help=False)      
        parser.add_argument("--epochs-G", default=2, type=int)
        parser.add_argument("--epochs-pred", default=1, type=int)
        parser.add_argument("--epochs-disc", default=1, type=int)
        parser.add_argument("--ema-beta", default=0.99, type=float, help="exponential moving average factor")
        parser.add_argument("--mu", default=1e-3, type=float)
        parser.add_argument('--reduction', default="sum", type=bool, choices=("sum", "mean"),
                            help="reduction for weighted loss")
        return parser
    
    @property
    def task(self) -> str:
        return "GAUR"