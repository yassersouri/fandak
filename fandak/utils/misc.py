import datetime
import subprocess


def get_git_commit_hash() -> str:
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
    print("[{}] {}".format(str(datetime.datetime.now()), the_thing))
