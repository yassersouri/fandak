from abc import ABC
from dataclasses import dataclass
from typing import List, Dict

import torch.nn as nn
from torch import Tensor
from yacs.config import CfgNode

from fandak.core.datasets import GeneralBatch
from fandak.utils.torch import GeneralDataClass


@dataclass(repr=False)
class GeneralLoss(GeneralDataClass):
    """
    I assume that there is always the `main` attribute which is the loss that is
    going to be used for backpropagation.
    """

    main: Tensor


@dataclass(repr=False)
class GeneralForwardOut(GeneralDataClass):
    """
    The general output of the forward pass.
    """

    pass


class Model(nn.Module, ABC):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg

    def get_params(self, original_lr: float) -> List[Dict]:
        params = [{"params": self.parameters(), "lr": original_lr}]
        return params

    # noinspection PyMethodMayBeStatic
    def get_backprop_loss(self, loss: GeneralLoss) -> Tensor:
        return loss.main

    def forward(self, batch: GeneralBatch) -> GeneralForwardOut:
        raise NotImplementedError

    def loss(self, batch: GeneralBatch, forward_out: GeneralForwardOut) -> GeneralLoss:
        raise NotImplementedError
