import random
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int = 1, fully_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def change_multiprocess_strategy(strategy: str = "file_system"):
    torch.multiprocessing.set_sharing_strategy(strategy)


def send_to_device(
    items: Union[Tensor, Tuple[Tensor, ...]], device
) -> Union[Tensor, Tuple[Tensor, ...]]:
    if type(items) is tuple:
        return tuple(map(lambda x: x.to(device), items))
    else:
        return items.to(device)


def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()
