from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
from yacs.config import CfgNode

from fandak import Dataset
from fandak.core.datasets import GeneralBatch


@dataclass
class Batch(GeneralBatch):
    x: Tensor  # [784] float
    y: Tensor  # [1] long


class MNISTClassification(Dataset):
    def __init__(self, cfg: CfgNode, train: bool):
        super().__init__(cfg)
        self.train = train
        call_args = {
            "root": cfg.dataset.root,
            "train": self.train,
            "download": True,
            "transform": transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        }
        if cfg.dataset.name == "fashion":
            self.mnist = FashionMNIST(**call_args)
        elif cfg.dataset.name == "digit":
            self.mnist = MNIST(**call_args)

    def __len__(self) -> int:
        return self.mnist.__len__()

    def __getitem__(self, item: int) -> Batch:
        img, target = self.mnist.__getitem__(item)
        batch = Batch(x=img.view(-1), y=torch.tensor(target).long())
        return batch

    @staticmethod
    def collate_fn(items: List[GeneralBatch]) -> GeneralBatch:
        pass
