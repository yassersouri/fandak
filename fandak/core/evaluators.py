from abc import ABC
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode

from fandak.core.datasets import Dataset, GeneralBatch
from fandak.core.models import Model, GeneralForwardOut
from fandak.utils.torch import GeneralDataClass


@dataclass(repr=False)
class GeneralEvaluatorResult(GeneralDataClass):
    pass


class Evaluator(ABC):
    def __init__(
        self,
        cfg: CfgNode,
        test_db: Dataset,
        model: Model,
        device: torch.device = torch.device("cpu"),
    ):
        self.cfg = cfg
        self.test_db = test_db
        self.model = model
        self.device = device

        # callback
        self.set_storage()

    def get_name(self) -> str:  # used in metrics
        return self.__class__.__name__

    def evaluate(self) -> GeneralEvaluatorResult:
        # callback
        self.on_start_eval()

        # prepare model
        self.model.to(self.device)
        self.model.eval()

        # create dataloader
        dataloader = self.create_dataloader()

        # perform evaluation
        with torch.no_grad():
            for batch in tqdm(dataloader):
                self._eval_1_batch(batch)

        # call back
        result = self.on_finish_eval()
        return result

    def _eval_1_batch(self, batch: GeneralBatch):
        batch.to(self.device)
        forward_out = self.model.forward(batch)
        self.batch_eval_calculation(batch, forward_out)

    def batch_eval_calculation(
        self, batch: GeneralBatch, forward_out: GeneralForwardOut
    ):
        raise NotImplementedError

    def create_dataloader(self) -> DataLoader:
        raise NotImplementedError

    def on_start_eval(self):
        pass

    def on_finish_eval(self) -> GeneralEvaluatorResult:
        raise NotImplementedError

    def set_storage(self):
        """
        create some storage to save intermediate evaluation results.
        """
        pass

    def reset_storage(self):
        """
        reset the storage. If you have some storage for the intermediate evaluation results,
        then they should be reset for every epoch evaluation.
        Here the default assumption is, if we recreate the storage, it will be reset.
        """
        self.set_storage()
