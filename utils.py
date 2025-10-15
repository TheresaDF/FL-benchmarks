from src.utils.constants import OPTIMIZERS
from omegaconf import DictConfig
import functools
import inspect 
import torch 

def get_client_optimizer_cls(config_args) -> type[torch.optim.Optimizer]:
    """Partial-init the client optimizer class with the config-provided args."""
    target_optimizer_cls: type[torch.optim.Optimizer] = OPTIMIZERS[
        config_args['optimizer']['name']
    ]
    keys_required = inspect.getfullargspec(target_optimizer_cls.__init__).args
    args_valid = {}
    for key, value in config_args['optimizer'].items():
        if key in keys_required:
            args_valid[key] = value

    optimizer_cls = functools.partial(target_optimizer_cls, **args_valid)
    return optimizer_cls