import random
from typing import Union, Collection

import numpy as np
import torch
from torch import Tensor

from fandak.utils.misc import is_listy


def set_seed(seed: int = 1, fully_deterministic: bool = False):
    """
    Sets the python, numpy and pytorch's ransom seeds.
    But this doesn't mean that you are fully deterministic.
    If you set fully_deterministic to True, your code will run slower, more deterministic, but not fully deterministic.
    You should also care about num_workers in your code and set them to 0.
    Also see: https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def change_multiprocess_strategy(strategy: str = "file_system"):
    """
    You might want to use this because of the following bug:
    https://github.com/pytorch/pytorch/issues/973
    """
    torch.multiprocessing.set_sharing_strategy(strategy)


def send_to_device(
    items: Collection[Tensor], device: torch.device
) -> Collection[Tensor]:
    """
    Send a single for a tuple of tensors
    """
    return tuple(map(lambda x: x.to(device), items))


def tensor_to_numpy(
    x: Union[Tensor, Collection[Tensor]]
) -> Union[np.ndarray, Collection[np.ndarray]]:
    if is_listy(x):
        return tuple(map(lambda t: tensor_to_numpy(t), x))
    else:
        return x.detach().cpu().numpy()
