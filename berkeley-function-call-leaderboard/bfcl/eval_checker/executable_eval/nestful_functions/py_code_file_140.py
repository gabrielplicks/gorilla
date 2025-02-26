import os

from typing import List



def get_all_filenames(directory: str) -> List[str]:

    """

    Retrieves a list of all file names in a given directory and its subdirectories.



    Args:

        directory: The path to the directory to traverse.



    Returns:

        A list of file names.

    """

    filenames = []



    for root, dirs, files in os.walk(directory):

        for filename in files:

            filenames.append(os.path.join(root, filename))



    return filenames

