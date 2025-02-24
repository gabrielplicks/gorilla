import os

from typing import List



def list_all_files(directory: str) -> List[str]:

    """Recursively traverses a file system directory and returns a list of all files, including their absolute paths.



    Args:

        directory: A string representing the directory path to traverse.



    Returns:

        A list of strings representing the absolute paths of all files in the directory and its subdirectories.

    """

    file_paths: List[str] = []

    for dirpath, dirnames, filenames in os.walk(directory):

        file_paths.extend([os.path.join(dirpath, filename) for filename in filenames])



    return file_paths

