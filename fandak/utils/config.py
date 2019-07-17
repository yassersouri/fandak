import argparse

from yacs.config import CfgNode


def default_parse_args() -> argparse.Namespace:
    """
    Default parsing of the input arguments for a CLI. Assumes usage of YACS.
    a single optional config file is requested.
    then all other config files can be set by --set option.
    """
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="optional config file", default=None, type=str
    )
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        help="set config keys",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def update_config(default_config: CfgNode) -> CfgNode:
    """
    This is useful for updating your config from CLI inputs, whether by a new config file or by --set.
    It will freeze the default config to prevent code smell.
    """
    args = default_parse_args()
    if args.cfg_file is not None:
        default_config.merge_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        default_config.merge_from_list(args.set_cfgs)

    default_config.freeze()

    return default_config
