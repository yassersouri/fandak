from abc import ABC
from typing import Tuple, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fandak.core.datasets import Dataset
from fandak.core.models import Model
from fandak.helpers import send_to_device


class Evaluator(ABC):
    def __init__(
        self,
        test_db: Dataset,
        model: Model,
        device: torch.device = torch.device("cpu"),
        writer: Optional[SummaryWriter] = None,
        fast: bool = True,
        batch_size: int = 1,
        num_workers: int = 1,
    ):
        self.test_db = test_db
        self.model = model
        self.device = device
        self.writer = writer
        self.test_iter_num = 0
        self.epoch_num = 0
        self.fast = fast
        self.test_mode = True
        self.batch_size = batch_size
        self.num_workers = num_workers

    def set_test_mode(self, test_mode: bool):
        self.test_mode = test_mode

    def set_writer(self, writer: SummaryWriter):
        self.writer = writer

    def set_epoch_number(self, epoch_num: int):
        self.epoch_num = epoch_num
        self.test_iter_num = (1 + self.epoch_num) * len(self.test_db)

    def evaluate(self):
        self.on_start_eval()
        dataloader = DataLoader(
            self.test_db,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_db.collate_fn,
        )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader):
                self.eval_1_batch(batch)
                self.test_iter_num += 1

        result = self.on_finish_eval()
        self.epoch_num += 1
        return result

    def eval_1_batch(self, batch: Tuple[Tensor, ...]):
        test_forward_input = self.model.prepare_forward_input_from_batch(
            batch, test_mode=self.test_mode
        )
        for k, v in test_forward_input.items():
            if v is not None:
                test_forward_input[k] = send_to_device(v, self.device)
        forward_out = self.model.forward(**test_forward_input)
        self.batch_eval_calculation(batch, forward_out)

    def on_start_eval(self):
        pass

    def batch_eval_calculation(
        self, batch: Tuple[Tensor, ...], forward_out: Tuple[Tensor, ...]
    ):
        raise NotImplementedError

    def on_finish_eval(self):
        raise NotImplementedError
