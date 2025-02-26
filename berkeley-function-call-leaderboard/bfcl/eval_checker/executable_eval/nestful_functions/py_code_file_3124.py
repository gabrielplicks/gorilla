import os

from typing import Union



def count_files_in_directory(directory_path: Union[str, os.PathLike]) -> int:

    """Counts the total number of files in a directory and all of its subdirectories.



    Args:

        directory_path: The path to the directory to count files in.



    Returns:

        The total number of files in the directory and its subdirectories.

    """

    total_files = 0

    for root, _, files in os.walk(directory_path):

        total_files += len(files)

    return total_files

