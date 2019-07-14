import torch

from fandak.utils import set_seed
from fandak.utils.config import update_config
from proj.config import get_config_defaults
from proj.datasets import MNISTClassification
from proj.evaluators import ValidationEvaluator
from proj.models import MLPModel
from proj.trainers import SimpleTrainer


def main():
    cfg = update_config(get_config_defaults())
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
        cfg, "first-experiment", train_db, model, device, evaluators
    )

    trainer.train()
    trainer.save_training()


if __name__ == "__main__":
    main()
