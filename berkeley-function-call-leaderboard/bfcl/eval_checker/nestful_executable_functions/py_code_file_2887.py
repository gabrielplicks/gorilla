import os

from typing import Tuple



def my_joinpath(path: str, root_dir: str) -> str:

    """Joins a path with the root directory.



    Args:

        path: The path to join with the root directory.

        root_dir: The root directory.



    Returns:

        The joined path.

    """

    if root_dir == '/':

        return path

    else:

        return os.path.join(root_dir, path)

