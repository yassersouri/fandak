from json import dumps as jdumps
from typing import List, Tuple

from torch.utils.data import Dataset as tDataset


class Dataset(tDataset):
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, item: int):
        raise NotImplementedError

    @staticmethod
    def collate_fn(items: List[Tuple]) -> Tuple:
        """
        Each dataset class should implement its own collate function.
        """
        raise NotImplementedError

    def json_repr(self) -> dict:
        raise NotImplementedError

    def __repr__(self) -> str:
        return jdumps(self.json_repr(), sort_keys=True)
