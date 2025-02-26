import os

from typing import List



def list_dir_members(root_dir: str) -> List[str]:

    """Lists all the filepaths of the members in a given directory, including subdirectories.

    Args:

        root_dir: The root directory to start the search from.

    Returns:

        A list of filepaths of the members in the given directory, including subdirectories.

    """

    filepaths = []

    for cur_dir, subdirs, files in os.walk(root_dir):

        for file in files:

            filepaths.append(os.path.join(cur_dir, file))

    return filepaths

