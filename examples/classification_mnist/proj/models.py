from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from yacs.config import CfgNode

from fandak import Model
from fandak.core.models import GeneralLoss
from proj.datasets import Batch


@dataclass(repr=False)
class ForwardOut:
    logits: Tensor  # [N x C]


class MLPModel(Model):
    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)
        self.x_size = cfg.dataset.x_size
        self.num_layers = cfg.model.mlp.num_layers
        self.num_hidden = cfg.model.mlp.num_hidden
        self.num_classes = cfg.dataset.num_classes
        layers = [
            nn.Linear(in_features=self.x_size, out_features=self.num_hidden),
            nn.ReLU(),
        ]
        for _ in range(1, self.num_layers):
            layers.append(
                nn.Linear(in_features=self.num_hidden, out_features=self.num_hidden)
            )
            layers.append(nn.ReLU())

        layers.append(
            nn.Linear(in_features=self.num_hidden, out_features=self.num_classes)
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, batch: Batch) -> ForwardOut:
        return ForwardOut(logits=self.layers(batch.x))

    def loss(self, batch: Batch, forward_out: ForwardOut) -> GeneralLoss:
        return GeneralLoss(main=F.cross_entropy(forward_out.logits, batch.y))
