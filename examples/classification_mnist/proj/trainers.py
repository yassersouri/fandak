from pathlib import Path
from typing import Optional

from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from fandak import Trainer
from fandak.core.trainers import Scheduler


class SimpleTrainer(Trainer):
    def figure_root(self) -> Path:
        return Path(self.cfg.system.root)

    def figure_optimizer(self) -> Optimizer:
        optimizer_name = self.cfg.trainer.optimizer.name
        learning_rate = self.cfg.trainer.optimizer.lr

        if optimizer_name == "Adam":
            return Adam(self.model.parameters(), lr=learning_rate)
        else:
            raise Exception("Invalid optimizer name (%s)" % optimizer_name)

    def figure_scheduler(self) -> Optional[Scheduler]:
        return None

    def figure_num_epochs(self) -> int:
        return self.cfg.trainer.num_epochs

    def create_train_dataloader(self) -> DataLoader:
        batch_size = self.cfg.trainer.batch_size
        num_workers = self.cfg.system.num_workers
        return DataLoader(
            self.train_db,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.train_db.collate_fn,
        )
