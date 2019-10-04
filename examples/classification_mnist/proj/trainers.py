from pathlib import Path
from typing import Optional, List, Any, Dict

from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from fandak import Trainer
from fandak.core.trainers import Scheduler
from .evaluators import TrainEvaluationResult


class SimpleTrainer(Trainer):
    def figure_root(self) -> Path:
        return Path(self.cfg.system.root)

    def figure_optimizer(self) -> Optimizer:
        optimizer_name = self.cfg.trainer.optimizer.name
        learning_rate = self.cfg.trainer.optimizer.lr

        if optimizer_name == "Adam":
            return Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)
        elif optimizer_name == "SGD":
            return SGD(self.model.parameters(), lr=learning_rate)
        else:
            raise Exception("Invalid optimizer name (%s)" % optimizer_name)

    def figure_scheduler(self, optimizer: Optimizer) -> Optional[Scheduler]:
        scheduler_name = self.cfg.trainer.scheduler.name
        scheduler_step = self.cfg.trainer.scheduler.step
        scheduler_steps = self.cfg.trainer.scheduler.steps
        gamma = self.cfg.trainer.scheduler.gamma

        if scheduler_name == "":
            return None
        elif scheduler_name == "Step":
            return StepLR(optimizer=optimizer, step_size=scheduler_step, gamma=gamma)
        elif scheduler_name == "MultiStep":
            return MultiStepLR(
                optimizer=optimizer, milestones=scheduler_steps, gamma=gamma
            )
        elif scheduler_name == "ReduceOnPlateau":
            return ReduceLROnPlateau(
                optimizer=optimizer,
                mode="min",
                factor=gamma,
                patience=2,
                threshold=1e-3,
                threshold_mode="abs",
                verbose=True,
            )
        else:
            raise Exception("Invalid scheduler name (%s)" % scheduler_name)

    def figure_scheduler_input(
        self, eval_results: List[TrainEvaluationResult]
    ) -> Dict[str, Any]:
        scheduler_name = self.cfg.trainer.scheduler.name

        if scheduler_name == "ReduceOnPlateau":
            return {"metrics": eval_results[0].average_loss}
        else:
            return {}

    def figure_accumulate_grad(self) -> int:
        return self.cfg.trainer.accumulate_grad_every

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
