from fandak.utils.config import update_config
from proj.config import get_config_defaults


def main():
    cfg = update_config(get_config_defaults())
    print(cfg)


if __name__ == "__main__":
    main()
