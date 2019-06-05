from abc import ABC
from json import dumps
from typing import List, Dict, Tuple

import torch.nn as nn
from torch import Tensor


class Model(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def get_params(self, original_lr: float) -> List[Dict]:
        params = [{"params": self.parameters(), "lr": original_lr}]
        return params

    def prepare_forward_input_from_batch(
        self, batch: Tuple[Tensor, ...], test_mode: bool = False
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def prepare_loss_input(
        self, batch: Tuple[Tensor, ...], forward_out: Tuple[Tensor, ...]
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def get_backprop_loss(self, losses: Tuple[Tensor, ...]) -> Tensor:
        if type(losses) == tuple:
            return losses[0]
        else:
            return losses

    def forward(self, **kwargs: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def loss(self, **kwargs: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return dumps(self.json_repr(), sort_keys=True, indent=4)

    def json_repr(self) -> dict:
        raise NotImplementedError
