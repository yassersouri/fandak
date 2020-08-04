import datetime
import sys
from abc import ABC
from pathlib import Path
from pickle import dump as pdump
from typing import Optional, Tuple, List, Union, Iterable, Dict, Any

import matplotlib.pylab as plt
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import MultiStepLR, StepLR, LambdaLR, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from yacs.config import CfgNode

from fandak.core.datasets import Dataset, GeneralBatch
from fandak.core.evaluators import Evaluator, GeneralEvaluatorResult
from fandak.core.models import Model, GeneralLoss, GeneralForwardOut
from fandak.utils.metrics import ScalarMetricCollection
from fandak.utils.misc import get_git_commit_hash, print_with_time
from fandak.utils.locks import lock

RUN_INFO_TEMPLATE = """Time: {time}  
Command: {command}  
Git hash: {hash}  

-----------------------------------------

{config}

"""

RUN_EXTRA_TEMPLATE = """-----------------------------------------

{extra}

"""

TORCH_EXT = "trc"
PICKLE_EXT = "pkl"
MODEL_FILE_NAME = "model"
OPTIMIZER_FILE_NAME = "optimizer"
SCHEDULER_FILE_NAME = "scheduler"

Scheduler = Union[MultiStepLR, StepLR, LambdaLR, ReduceLROnPlateau]


class Trainer(ABC):
    main_loss_metric_name = "main/loss"
    eval_metric_name_format = "eval/{}"

    def __init__(
        self,
        cfg: CfgNode,
        exp_name: str,
        train_db: Dataset,
        model: Model,
        device: torch.device = torch.device("cpu"),
        evaluators: Optional[Union[Iterable[Evaluator], Evaluator]] = None,
    ):
        self.cfg = cfg
        self.exp_name = exp_name
        self.train_db = train_db
        self.model = model
        self.device = device
        if evaluators is not None:
            if not isinstance(evaluators, Iterable):
                evaluators = [evaluators]
        else:
            evaluators = []
        self.evaluators = evaluators  # type: Iterable[Evaluator]

        self.save_every = 1
        # if you use reduce on plateau scheduler then this should be 1
        # because each epoch you need to have the evaluation value
        self.eval_every = 1

        self.root = self.figure_root().expanduser()
        self.optimizer = self.figure_optimizer()
        self.scheduler = self.figure_scheduler(self.optimizer)
        self.clip_grad_norm = self.figure_clip_grad_norm()

        self.tb_root = self.root / "TB"
        self.model.to(device)
        self.epoch_num = 0
        self.iter_num = 0

        self.experiment_folder = self.root / self.exp_name
        self.experiment_folder.mkdir(exist_ok=True, parents=True)
        self.run_number = self._figure_run_number()
        self._set_run_folder()
        self.update_trainer_using_config()

    def _set_run_folder(self):
        self.run_folder = self.experiment_folder / str(self.run_number)
        self.tb_folder = self.tb_root / self.exp_name / str(self.run_number)

    # noinspection PyMethodMayBeStatic
    def extra_info_for_run(self) -> Optional[dict]:
        return None

    def _save_info_of_run(self):
        extra_info = self.extra_info_for_run()

        # fixme: assuming self.cfg has a `.dump` function.
        config_dump = self.cfg.dump()

        # adding 4 spaces at the beginning of each line
        # this is because now tensorboard's text section can visualize it as code block
        config_dump = "\n".join([f"    {l}" for l in config_dump.split("\n")])

        final_value = RUN_INFO_TEMPLATE

        result = final_value.format(
            time=str(datetime.datetime.now()),
            command=" ".join(sys.argv),
            hash=get_git_commit_hash(),
            config=config_dump,
        )

        if extra_info:
            result += RUN_EXTRA_TEMPLATE.format(extra=extra_info)

        with open(self.run_folder / Path("info.md"), "w") as f:
            f.write(result)

        # Add the info to tensorboard for easier observation.
        self.writer.add_text("info", result)

        config_path = self.run_folder / "config.yaml"
        with open(config_path, "w") as f:
            f.write(self.cfg.dump())

    def save_run_report(
        self, report: str, name: str = "report", extension: str = "txt"
    ):
        with open(self.run_folder / Path("%s.%s" % (name, extension)), "w") as f:
            f.write(report)

    def save_fig(self, fig: plt.Figure, name: str = "figure"):
        fig.savefig(self.run_folder / ("%s.png" % name))

    def _mark_the_run(self):
        self.writer = SummaryWriter(str(self.tb_folder))
        self.run_folder.mkdir(exist_ok=True, parents=True)
        self._save_info_of_run()
        self.metrics = self.create_metrics()

    def train(self):
        num_epochs = self.figure_num_epochs()
        self._mark_the_run()
        print_with_time("Training for run number: {:d}".format(self.run_number))
        epoch_range_start = self.epoch_num
        epoch_range = range(epoch_range_start, epoch_range_start + num_epochs)

        # callback
        self.on_start_training(num_epochs)

        for epoch_num in epoch_range:
            self.epoch_num = epoch_num

            # callback
            self.on_start_epoch(epoch_num)

            # resetting metrics
            for n, m in self.metrics.items():
                if self.metrics[n].report_average:
                    m.reset_values()

            # training for 1 epoch
            dataloader = self.create_train_dataloader()
            self.train_1_epoch(epoch_num, dataloader)

            # saving
            if (epoch_num + 1) % self.save_every == 0:
                self.save_training()

            # evaluation
            eval_results = []
            if (epoch_num + 1) % self.eval_every == 0:
                for evaluator in self.evaluators:
                    eval_results.append(evaluator.evaluate())
                    evaluator.reset_storage()
                self.track_end_of_epoch_metrics(eval_results, epoch_num)

            # scheduler
            if self.scheduler is not None:
                # noinspection PyArgumentList
                self.scheduler.step(**self.figure_scheduler_input(eval_results))

            # callback
            self.on_finish_epoch(epoch_num)

        # callback
        self.on_finish_training(num_epochs)

    def train_1_epoch(self, epoch_number: int, dataloader: DataLoader):
        print_with_time("Training epoch %d ..." % (epoch_number + 1))
        self.model.to(self.device)
        self.model.train()

        for batch in tqdm(dataloader):
            # callback
            self.on_start_batch(self.iter_num, batch)

            # train for 1 batch
            batch_loss, forward_out = self._train_1_batch(self.iter_num, batch)

            # update metrics for 1 batch
            self.track_training_metrics(batch, forward_out, batch_loss, self.iter_num)

            # call back
            self.on_finish_batch(self.iter_num, batch, forward_out, batch_loss)
            self.iter_num += 1

    # noinspection PyUnusedLocal
    def _train_1_batch(
        self, iter_num: int, batch: GeneralBatch
    ) -> Tuple[GeneralLoss, GeneralForwardOut]:
        # callback
        self.on_start_batch(self.iter_num, batch)

        # FIXME: this might be slow depending on the config system
        accumulate_grad_every = self.figure_accumulate_grad()
        if accumulate_grad_every is None:
            accumulate_grad_every = 1

        # TODO: move to the end.
        # TODO: move to infinitely flexible callbacks. like fastai v2.
        # initial setup
        if iter_num % accumulate_grad_every == 0:
            self.optimizer.zero_grad()
        batch.to(self.device)

        # forward pass
        forward_out = self.model.forward(batch)
        loss = self.model.loss(batch, forward_out)

        the_loss = loss.main / accumulate_grad_every

        # backward pass
        the_loss.backward()

        # optional gradient clipping
        if self.clip_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

        # optimizer step
        if iter_num % accumulate_grad_every == (accumulate_grad_every - 1):
            self.optimizer.step()

        # callback
        self.on_finish_batch(self.iter_num, batch, forward_out, loss)

        return loss, forward_out

    def save_training(self):
        print_with_time("Saving model ...")
        self.checkpoint_this(MODEL_FILE_NAME, self.model.state_dict(), torch_save=True)
        self.checkpoint_this(
            OPTIMIZER_FILE_NAME, self.optimizer.state_dict(), torch_save=True
        )
        if self.scheduler is not None:
            self.checkpoint_this(SCHEDULER_FILE_NAME, self.scheduler, torch_save=True)

    def load_training(self, run: int, epoch: int):
        """
        todo: make epoch optional by looking for the largest value for it.
        """
        assert epoch >= 0
        self.run_number = run
        self.epoch_num = epoch - 1  # todo: is this correct?

        # this is not a good thing to do, we need a way to save the iter_num on disk maybe.
        self.iter_num = (1 + self.epoch_num) * len(self.train_db)

        self._set_run_folder()
        check_pointing_folder = self._get_checkpointing_folder()

        model_file_name = check_pointing_folder / (
            "%s.%s" % (MODEL_FILE_NAME, TORCH_EXT)
        )
        self.model.load_state_dict(torch.load(model_file_name))

        optimizer_file_name = check_pointing_folder / (
            "%s.%s" % (OPTIMIZER_FILE_NAME, TORCH_EXT)
        )
        self.optimizer.load_state_dict(torch.load(optimizer_file_name))

        if self.scheduler is not None:
            scheduler_file_name = check_pointing_folder / (
                "%s.%s" % (SCHEDULER_FILE_NAME, TORCH_EXT)
            )
            self.scheduler = torch.load(scheduler_file_name)

    def _get_checkpointing_folder(self) -> Path:
        return self.run_folder / str(self.epoch_num + 1)

    def checkpoint_this(self, name: str, thing, torch_save: bool = False):
        check_pointing_folder = self._get_checkpointing_folder()
        check_pointing_folder.mkdir(exist_ok=True, parents=True)
        if torch_save:
            file_name = check_pointing_folder / Path("%s.%s" % (name, TORCH_EXT))
            torch.save(thing, file_name)
        else:
            file_name = check_pointing_folder / Path("%s.%s" % (name, PICKLE_EXT))
            with open(file_name, "wb") as f:
                pdump(thing, f)

    def on_start_training(self, num_epochs: int):
        pass

    def on_start_epoch(self, epoch_num: int):
        pass

    def on_start_batch(self, iter_num: int, batch: GeneralBatch):
        pass

    def on_finish_batch(
        self,
        iter_num: int,
        batch: GeneralBatch,
        forward_out: GeneralForwardOut,
        loss: GeneralLoss,
    ):
        pass

    def on_finish_epoch(self, epoch_num: int):
        pass

    def on_finish_training(self, num_epochs: int):
        pass

    def _figure_run_number(self) -> int:
        # fixme: this is not thread safe!
        max_run = 0
        with lock(f"run_number.{str(self.experiment_folder)}"):
            for f in self.experiment_folder.iterdir():
                if f.is_dir():
                    try:
                        f = int(str(f.name))
                    except ValueError:
                        continue
                    if f > max_run:
                        max_run = f

        return max_run + 1

    def figure_root(self) -> Path:
        raise NotImplementedError

    def figure_optimizer(self) -> Optimizer:
        raise NotImplementedError

    def figure_scheduler(self, optimizer: Optimizer) -> Optional[Scheduler]:
        raise NotImplementedError

    def figure_accumulate_grad(self) -> Optional[int]:
        return None

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def figure_scheduler_input(
        self, eval_results: List[GeneralEvaluatorResult]
    ) -> Dict[str, Any]:
        """
        If you want to use the on plateau lr_scheduler for example, then return something like
        {"metrics": eval_results[0].eval_loss.item()}.
        You should implement it as you like.
        The default implementation will return empty dictionary.
        """
        return {}

    def figure_num_epochs(self) -> int:
        raise NotImplementedError

    def create_train_dataloader(self) -> DataLoader:
        """
        If you want to overfit your model, you have to implement it here.
        """
        raise NotImplementedError

    def update_trainer_using_config(self):
        """
        This is called at the end of init.
        So change the implementation if you like.
        """
        pass

    # noinspection PyMethodMayBeStatic
    def figure_clip_grad_norm(self) -> Optional[float]:
        """
        TODO: Think about removing this and adding it as an optional hook.
        Return a positive value if you want gradient clipping.
        Return None is you don't want gradient clipping.
        """
        return None

    def create_metrics(self) -> Dict[str, ScalarMetricCollection]:
        """
        By default we create a report average metric for the main loss.
        Also  a non-report average metric for each evaluator.
        """
        default_loss_metric = ScalarMetricCollection(
            writer=self.writer,
            root=self.run_folder,
            base_name="loss",
            print_each_iter=False,
        )
        metrics = {self.main_loss_metric_name: default_loss_metric}

        for i, evaluator in enumerate(self.evaluators):
            metrics[
                self.eval_metric_name_format.format(i + 1)
            ] = ScalarMetricCollection(
                writer=self.writer,
                root=self.run_folder,
                base_name=evaluator.get_name(),
                print_each_iter=True,
                report_average=False,
            )

        return metrics

    # noinspection PyUnusedLocal
    def track_training_metrics(
        self,
        batch: GeneralBatch,
        forward_out: GeneralForwardOut,
        loss: GeneralLoss,
        iter_num: int,
    ):
        """
        By default we will update the loss.
        """
        self.metrics[self.main_loss_metric_name].add_value(dc_value=loss, step=iter_num)

    def track_end_of_epoch_metrics(
        self, eval_results: List[GeneralEvaluatorResult], epoch_num: int
    ):
        """
        By default we will update the loss for average reporting. Also if any evaluator is provided
        they will also be updated.
        """
        # for the main loss
        self.metrics[self.main_loss_metric_name].epoch_finished(epoch_num=epoch_num)

        # for each evaluator
        for i, (evaluator, eval_result) in enumerate(
            zip(self.evaluators, eval_results)
        ):
            # noinspection PyTypeChecker
            self.metrics[self.eval_metric_name_format.format(i + 1)].add_value(
                dc_value=eval_result, step=(epoch_num + 1)
            )
            self.metrics[self.eval_metric_name_format.format(i + 1)].save()
