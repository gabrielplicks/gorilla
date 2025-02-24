import os

from typing import List



def file_path(filename: str, directories: List[str]) -> str:

    """Returns a string that represents the full path of the file.

    Args:

        filename: The name of the file.

        directories: A list of subdirectories.

    """

    path = os.path.join(*directories, filename)

    return path

