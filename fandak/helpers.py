import datetime
import random
import subprocess
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int, fully_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fully_deterministic:
        torch.backends.cudnn.deterministic = True


def standard_repr(class_name: str, list_props: List[Tuple[str, str]]) -> str:
    result = "{cn}({props})"
    props = "-".join("%s:%s" % (k, v) for k, v in list_props)

    return result.format(cn=class_name, props=props)


def send_to_device(
        items: Union[Tensor, Tuple[Tensor, ...]], device
) -> Union[Tensor, Tuple[Tensor, ...]]:
    if type(items) is tuple:
        return tuple(map(lambda x: x.to(device), items))
    else:
        return items.to(device)


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
        )
    except subprocess.CalledProcessError:
        # this is probably not in a git repo or git is not installed.
        return ""


def tensor_to_numpy(x: Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def tensors_to_numpys(
        x: Union[Tensor, Tuple[Tensor, ...]]
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    if type(x) is not tuple:
        return tensor_to_numpy(x)
    else:
        return tuple(map(lambda i: tensor_to_numpy(i), x))


def change_multiprocess_strategy():
    torch.multiprocessing.set_sharing_strategy("file_system")


def print_with_time(the_thing: str):
    print("[{}] {}".format(str(datetime.datetime.now()), the_thing))
