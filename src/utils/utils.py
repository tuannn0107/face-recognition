from pathlib import Path


def get_file_path(pure_path):
    res = Path(pure_path).parent
    return res


def get_parent_path(pure_path):
    return get_file_path(pure_path).parent

def get_root_path():
    return Path(__file__).parent.parent


