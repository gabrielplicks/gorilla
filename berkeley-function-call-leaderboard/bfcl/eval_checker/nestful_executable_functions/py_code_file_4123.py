import os



def safe_path_concat(path1: str, path2: str) -> str:

    """Safely concatenates paths on Windows and Linux/Unix systems.

    Args:

        path1: The first path.

        path2: The second path.

    """

    return os.path.join(path1, path2)

