from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader

from fandak import Evaluator
from fandak.core.evaluators import GeneralEvaluatorResult
from fandak.utils import print_with_time
from fandak.utils.torch import tensor_to_numpy
from proj.datasets import Batch
from proj.models import ForwardOut


@dataclass
class TrainEvaluationResult(GeneralEvaluatorResult):
    accuracy: float


class ValidationEvaluator(Evaluator):
    # noinspection PyAttributeOutsideInit
    def set_storage(self):
        self.preds = []
        self.targets = []

    def on_finish_eval(self) -> TrainEvaluationResult:
        preds = np.array(self.preds)
        targets = np.array(self.targets)

        accuracy = sum(preds == targets) / len(preds)

        print_with_time("Evaluation accuracy: %1.6f" % accuracy)

        return TrainEvaluationResult(accuracy=accuracy)

    def batch_eval_calculation(self, batch: Batch, forward_out: ForwardOut):
        preds = tensor_to_numpy(forward_out.logits.argmax(dim=1))
        targets = tensor_to_numpy(batch.y)

        for i in range(len(targets)):
            self.preds.append(preds[i])
            self.targets.append(targets[i])

    def create_dataloader(self) -> DataLoader:
        batch_size = self.cfg.evaluator.eval_batch_size
        num_workers = self.cfg.system.num_workers
        return DataLoader(
            self.test_db,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.test_db.collate_fn,
        )