import click


def common_config(function):
    function = click.option(
        "--cfg",
        "file_configs",
        type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
        multiple=True,
    )(function)
    function = click.option("--set", "set_configs", type=(str, str), multiple=True)(
        function
    )

    return function
