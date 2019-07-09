import datetime
import subprocess
from typing import Any


def get_git_commit_hash() -> str:
    """
    Returns the hash of the HEAD commit. Or an empty string.
    Can be used for logging purposes.
    """
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        # this is probably not in a git repo or git is not installed.
        return ""


def print_with_time(the_thing: str):
    """
    Print an string with time.
    """
    print("[{}] {}".format(str(datetime.datetime.now()), the_thing))


def is_listy(x: Any) -> bool:
    """
    Grabbed this from fast.ai
    """
    return isinstance(x, (tuple, list))
