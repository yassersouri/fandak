from torch.utils.data import DataLoader

from fandak.utils.config import update_config
from proj.config import get_config_defaults
from proj.datasets import MNISTClassification, Batch


def main():
    cfg = update_config(get_config_defaults())
    print(cfg)

    train_db = MNISTClassification(cfg, train=True)
    # test_db = MNISTClassification(cfg, train=False)

    train_loader = DataLoader(train_db, batch_size=2, collate_fn=Batch.default_collate)

    x = train_loader.__iter__().__next__()
    print(x)


if __name__ == "__main__":
    main()
