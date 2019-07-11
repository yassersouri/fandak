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
_C.dataset.x_size = 784
_C.dataset.num_classes = 10

_C.model = CN()
_C.model.name = "MLP"  # MLP

_C.model.mlp = CN()
_C.model.mlp.num_layers = 1
_C.model.mlp.num_hidden = 128
_C.model.mlp.last_drop = 0.0


def get_config_defaults():
    return _C.clone()
