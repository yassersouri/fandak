from dataclasses import dataclass
from typing import List

from torch import Tensor
from torch.utils.data import Dataset as tDataset

# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import default_collate
from yacs.config import CfgNode

from fandak.utils.torch import GeneralDataClass


@dataclass(repr=False)
class GeneralBatch(GeneralDataClass):
    @classmethod
    def default_collate(cls, items: List["GeneralBatch"]) -> "GeneralBatch":
        """
        Uses default_collate from `torch.utils.data.Dataloader.default_collate`
        for every attribute of the type to collate them together.
        """
        values = {}
        for attr_name in items[0].get_attribute_names():
            attrs = [getattr(i, attr_name) for i in items]
            values[attr_name] = default_collate(attrs)

        # noinspection PyArgumentList
        return cls(**values)


class Dataset(tDataset):
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.cfg = cfg

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item: int) -> GeneralBatch:
        raise NotImplementedError

    def collate_fn(self, items: List[GeneralBatch]) -> GeneralBatch:
        """
        Each dataset class should implement its own collate function.
        Could be as simple as `return GeneralBatch.default_collate(items)`
        """
        raise NotImplementedError
