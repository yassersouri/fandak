from torch.utils.data import DataLoader

from fandak.utils.config import update_config
from proj.config import get_config_defaults
from proj.datasets import MNISTClassification, Batch
from proj.models import MLPModel


def main():
    cfg = update_config(get_config_defaults())
    print(cfg)

    train_db = MNISTClassification(cfg, train=True)
    # test_db = MNISTClassification(cfg, train=False)

    train_loader = DataLoader(train_db, batch_size=2, collate_fn=Batch.default_collate)
    batch = train_loader.__iter__().__next__()

    if cfg.model.name == "MLP":
        model = MLPModel(cfg)
    else:
        raise Exception("Invalid model name (%s)" % cfg.model.name)

    forward_out = model.forward(batch)
    loss = model.loss(batch, forward_out)

    model.get_backprop_loss(loss).backward()

    print(batch)


if __name__ == "__main__":
    main()
