import os

from typing import Union



def get_depth(path: Union[str, bytes]) -> int:

    """Calculates the hierarchical depth of a path.



    Args:

        path: The path to calculate the depth of.



    Raises:

        ValueError: If the path does not exist.

    """

    if not os.path.exists(path):

        raise ValueError("Path does not exist")



    normalized_path = os.path.normpath(path)

    depth = normalized_path.count(os.path.sep) - 1



    return depth

