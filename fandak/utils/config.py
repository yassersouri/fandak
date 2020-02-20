from typing import List

from yacs.config import CfgNode


def update_config(
    default_config: CfgNode,
    file_configs: List[str],
    set_configs: List[str],
    freeze: bool = True,
) -> CfgNode:
    """
    This is useful for updating your config from CLI inputs, whether by new config files or by --set.
    It will freeze the default config to prevent code smell.
    """
    cfg = default_config
    # updating config from file
    for fc in file_configs:
        cfg.merge_from_file(fc)
    # updating config from set
    for sc in set_configs:
        cfg.merge_from_list(list(sc))

    if freeze:
        cfg.freeze()

    return cfg
