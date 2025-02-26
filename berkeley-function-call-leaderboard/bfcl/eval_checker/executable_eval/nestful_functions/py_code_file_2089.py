import os

from typing import List



def list_files_and_directories(path: str) -> List[str]:

    """Returns a list of all files and directories under the given path, including the path itself.



    Args:

        path: The path to the directory.



    Returns:

        A list of paths.

    """

    file_list = []

    for root, dirs, files in os.walk(path):

        for file in files:

            file_list.append(os.path.join(root, file))

        for dir in dirs:

            file_list.append(os.path.join(root, dir))

    return file_list

