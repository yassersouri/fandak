from typing import List

import click
import torch

from fandak.utils import common_config
from fandak.utils import set_seed
from fandak.utils.config import update_config
from proj.config import get_config_defaults
from proj.datasets import MNISTClassification
from proj.evaluators import ValidationEvaluator
from proj.models import MLPModel
from proj.trainers import SimpleTrainer


@click.command()
@common_config
def main(file_configs: List[str], set_configs: List[str]):

    cfg = update_config(
        default_config=get_config_defaults(),
        file_configs=file_configs,
        set_configs=set_configs,
    )
    print(cfg)

    set_seed(cfg.system.seed)
    device = torch.device(cfg.system.device)

    train_db = MNISTClassification(cfg, train=True)
    test_db = MNISTClassification(cfg, train=False)

    if cfg.model.name == "MLP":
        model = MLPModel(cfg)
    else:
        raise Exception("Invalid model name (%s)" % cfg.model.name)

    evaluators = [ValidationEvaluator(cfg, test_db, model, device)]
    trainer = SimpleTrainer(
        cfg, cfg.experiment_name, train_db, model, device, evaluators
    )

    trainer.train()
    trainer.save_training()


if __name__ == "__main__":
    main()
