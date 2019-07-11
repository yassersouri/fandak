import torch
from yacs.config import CfgNode as CN

_C = CN()

_C.system = CN()
_C.system.device = "cuda" if torch.cuda.is_available() else "cpu"
_C.system.num_workers = 1
_C.system.seed = 1

_C.dataset = CN()
_C.dataset.name = "digit"  # digit, fashion
_C.dataset.root = "/home/souri/temp/"


def get_config_defaults():
    return _C.clone()
