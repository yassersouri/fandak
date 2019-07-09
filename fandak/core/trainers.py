import datetime
import os
import sys
from abc import ABC
from json import dump as jdump
from json import dumps as jdumps
from pickle import dump as pdump
from typing import Optional, Tuple, List, Union

import matplotlib.pylab as plt
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fandak.core.datasets import Dataset
from fandak.core.evaluators import Evaluator
from fandak.core.models import Model
from fandak.utils.torch import send_to_device
from fandak.utils.misc import get_git_commit_hash, print_with_time

TORCH_EXT = "trc"
PICKLE_EXT = "pkl"
MODEL_FILE_NAME = "model"
OPTIMIZER_FILE_NAME = "optimizer"
SCHEDULER_FILE_NAME = "scheduler"


class Trainer(ABC):
    def __init__(
        self,
        exp_name: str,
        results_root: str,
        train_db: Dataset,
        model: Model,
        optimizer: Optimizer,
        scheduler: Optional[MultiStepLR] = None,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        seed=0,
        evaluators: Optional[Union[List[Evaluator], Evaluator]] = None,
        clip_grad_norm: Optional[float] = None,
        fast: bool = True,
        num_workers: int = 1,
        **kwargs
    ):
        self.exp_name = exp_name
        self.train_db = train_db
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.fast = fast
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.results_root = results_root
        self.tb_root = os.path.join(self.results_root, "TB")

        if evaluators is not None:
            if type(evaluators) is not list:
                evaluators = [evaluators]
        self.evaluators = evaluators  # type: Optional[List[Evaluator]]

        self.clip_grad_norm = clip_grad_norm

        self.model.to(device)

        self.epoch_num = -1  # before training
        self.iter_num = 0
        self.test_iter_num = 0
        self.minibatch_losses = None
        self.epoch_losses = []
        self.evaluation_values = []

        self.experiment_folder = os.path.join(self.results_root, self.exp_name)
        os.makedirs(self.experiment_folder, exist_ok=True)
        self.run_number = self._figure_run_number()
        self.additional_tb_name = self.__repr__()
        self._set_run_folder()

    def _set_run_folder(self):
        self.run_folder = os.path.join(self.experiment_folder, "%d" % self.run_number)
        self.tb_folder = os.path.join(
            self.tb_root, self.exp_name, "%d" % self.run_number
        )
        self.writer = SummaryWriter(self.tb_folder, comment=self.additional_tb_name)

        if self.evaluators is not None:
            for ev in self.evaluators:
                ev.set_writer(self.writer)

    def _get_training_repr(self) -> str:
        # TODO: change to json
        # noinspection PyListCreation
        result = ["batch_size(%d)" % self.batch_size]
        optimizer_props = [("lr", str(self.optimizer.defaults["lr"]))]
        if "weight_decay" in self.optimizer.defaults:
            optimizer_props += [("wd", str(self.optimizer.defaults["weight_decay"]))]
        if "momentum" in self.optimizer.defaults:
            optimizer_props += [("momentum", str(self.optimizer.defaults["momentum"]))]
        result.append(standard_repr(self.optimizer.__class__.__name__, optimizer_props))
        if self.scheduler is not None:
            result.append(
                standard_repr(
                    "scheduler",
                    [
                        ("steps", str(self.scheduler.milestones)),
                        ("gamma", str(self.scheduler.gamma)),
                    ],
                )
            )

        return "-".join(result)

    # noinspection PyMethodMayBeStatic
    def extra_info_for_run(self) -> Optional[dict]:
        return None

    def _save_info_of_run(self):
        # todo: move away from string to dictionary.
        info = {
            "exp_name": self.exp_name,
            "train_db": self.train_db.json_repr(),
            "model": self.model.json_repr(),
            "training": self._get_training_repr(),
            "git_hash": get_git_commit_hash(),
            "datetime": str(datetime.datetime.now()),
            "random-seed": str(self.seed),
            "call": " ".join(sys.argv),
            "clip_grad_norm": self.clip_grad_norm,
        }

        extra_info = self.extra_info_for_run()
        if extra_info is not None:
            info.update(extra_info)

        with open(os.path.join(self.run_folder, "info.json"), "w") as f:
            jdump(info, f, sort_keys=True, indent=4)

        # Add the info to tensorboard for easier observation.
        self.writer.add_text("info.json", jdumps(info, sort_keys=True, indent=4))

    def save_run_report(
        self, report: str, name: str = "report", extension: str = "txt"
    ):
        with open(os.path.join(self.run_folder, "%s.%s" % (name, extension)), "w") as f:
            f.write(report)

    def save_fig(self, fig: plt.Figure, name: str = "figure"):
        fig.savefig(os.path.join(self.run_folder, "%s.png" % name))

    def _mark_the_run(self):
        os.makedirs(self.run_folder, exist_ok=True)
        self._save_info_of_run()

    def train(self, num_epochs: int):
        self._mark_the_run()  # todo: is this the best place? my concern is with resuming, etc.
        print_with_time("Training for run number: {:d}".format(self.run_number))
        epoch_range_start = 1 + self.epoch_num
        epoch_range = range(epoch_range_start, epoch_range_start + num_epochs)
        self.on_start_training(num_epochs)
        for epoch_num in epoch_range:
            self.epoch_losses = []
            if self.scheduler is not None:
                self.scheduler.step(epoch_num)
            # todo: add parameters for shuffle
            dataloader = DataLoader(
                self.train_db,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=self.train_db.collate_fn,
            )
            self.on_start_epoch(epoch_num)
            for batch in tqdm(dataloader):
                self.model.train()

                losses = self.train_1_batch(batch)
                self.minibatch_losses = losses
                self.epoch_losses.append(losses)

                self.iter_num += 1
            self.epoch_num = epoch_num
            self.on_finish_epoch(epoch_num, self.epoch_losses)
            if self.evaluators is not None and len(self.evaluators) > 0:
                results = []
                for evaluator in self.evaluators:
                    results.append(evaluator.evaluate())
                self.evaluation_values.append(results)
            print_with_time("epoch %d done." % epoch_num)

    def train_1_batch(self, batch: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        self.on_start_batch(self.iter_num, batch)
        self.optimizer.zero_grad()

        forward_input = self.model.prepare_forward_input_from_batch(batch)
        for k, v in forward_input.items():
            forward_input[k] = send_to_device(v, self.device)
        forward_out = self.model.forward(**forward_input)
        loss_input = self.model.prepare_loss_input(batch, forward_out)
        for k, v in loss_input.items():
            loss_input[k] = send_to_device(v, self.device)
        losses = self.model.loss(**loss_input)
        loss = self.model.get_backprop_loss(losses)

        self.writer.add_scalar("loss/main", loss.item(), global_step=self.iter_num)
        loss.backward()

        # optional gradient clipping
        if self.clip_grad_norm is not None:
            for param_set in self.model.get_params(0):
                clip_grad_norm_(param_set["params"], max_norm=self.clip_grad_norm)

        # noinspection PyArgumentList
        self.optimizer.step()

        self.on_finish_batch(self.iter_num, batch, losses, forward_out)
        return losses

    def __repr__(self) -> str:
        return "train_db({tdb})-model({m})-training({trng})".format(
            tdb=self.train_db.__repr__(),
            m=self.model.__repr__(),
            trng=self._get_training_repr(),
        )

    def save_training(self):
        self.checkpoint_this(MODEL_FILE_NAME, self.model.state_dict(), torch_save=True)
        self.checkpoint_this(
            OPTIMIZER_FILE_NAME, self.optimizer.state_dict(), torch_save=True
        )
        if self.scheduler is not None:
            self.checkpoint_this(SCHEDULER_FILE_NAME, self.scheduler, torch_save=True)

    def load_training(self, run: int, epoch: int):
        """
        epoch starts from 0, so, epoch=0 means load training from after epoch 0.
        todo: make epoch optional by looking for the largest value for it.
        todo: this starts from 0 thing is unreasonable.
        """
        assert epoch >= 0
        self.run_number = run
        self.epoch_num = epoch
        if self.evaluators is not None:
            for ev in self.evaluators:
                ev.set_epoch_number(self.epoch_num)

        # fixme: this is not a good thing to do, we need a way to save the iter_num on disk maybe.
        self.iter_num = (1 + self.epoch_num) * len(self.train_db)

        self._set_run_folder()
        check_pointing_folder = self._get_checkpointing_folder()

        model_file_name = os.path.join(
            check_pointing_folder, "%s.%s" % (MODEL_FILE_NAME, TORCH_EXT)
        )
        self.model.load_state_dict(torch.load(model_file_name))

        optimizer_file_name = os.path.join(
            check_pointing_folder, "%s.%s" % (OPTIMIZER_FILE_NAME, TORCH_EXT)
        )
        self.optimizer.load_state_dict(torch.load(optimizer_file_name))

        if self.scheduler is not None:
            scheduler_file_name = os.path.join(
                check_pointing_folder, "%s.%s" % (SCHEDULER_FILE_NAME, TORCH_EXT)
            )
            self.scheduler = torch.load(scheduler_file_name)

    def _get_checkpointing_folder(self) -> str:
        return os.path.join(self.run_folder, str(self.epoch_num))

    def checkpoint_this(self, name: str, thing, torch_save: bool = False):
        check_pointing_folder = self._get_checkpointing_folder()
        os.makedirs(check_pointing_folder, exist_ok=True)
        if torch_save:
            file_name = os.path.join(check_pointing_folder, "%s.%s" % (name, TORCH_EXT))
            torch.save(thing, file_name)
        else:
            file_name = os.path.join(
                check_pointing_folder, "%s.%s" % (name, PICKLE_EXT)
            )
            with open(file_name, "wb") as f:
                pdump(thing, f)

    def on_start_training(self, num_epochs: int):
        pass

    def on_start_epoch(self, epoch_num: int):
        pass

    def on_start_batch(self, iter_num: int, batch: Tuple[Tensor, ...]):
        pass

    def on_finish_batch(
        self,
        iter_num: int,
        batch: Tuple[Tensor, ...],
        losses: Tuple[Tensor, ...],
        forward_out: Tuple[Tensor, ...],
    ):
        pass

    def on_finish_epoch(self, epoch_num: int, epoch_losses: List[Tuple[Tensor, ...]]):
        pass

    def on_finish_training(self, num_epochs: int):
        pass

    def _figure_run_number(self) -> int:
        # fixme: this is not thread safe!
        max_run = 0
        for f in os.listdir(self.experiment_folder):
            if os.path.isdir(os.path.join(self.experiment_folder, f)):
                try:
                    f = int(f)
                except ValueError:
                    continue
                if f > max_run:
                    max_run = f

        return max_run + 1
