import random
from typing import Union, Collection, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler

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


class OverfitSampler(Sampler):
    def __init__(self, main_source, indices):
        super().__init__(main_source)
        self.main_source = main_source
        self.indices = indices

        main_source_len = len(self.main_source)

        how_many = int(round(main_source_len / len(self.indices)))
        self.to_iter_from = []
        for _ in range(how_many):
            self.to_iter_from.extend(self.indices)

    def __iter__(self):
        return iter(self.to_iter_from)

    def __len__(self):
        return len(self.main_source)


class GeneralDataClass:
    def get_attribute_names(self) -> List[str]:
        return [
            a
            for a in dir(self)
            if not a.startswith("__") and not callable(getattr(self, a))
        ]

    def to(self, device: torch.device):
        for attr_name in self.get_attribute_names():
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))

    def item(self):
        """
        This is like calling `.item()` for all of the attributes and setting their values to it.
        Will raise error if more than a single item is inside the tensor.
        """
        for attr_name in self.get_attribute_names():
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.item())