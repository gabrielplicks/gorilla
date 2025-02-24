import os

from typing import Union



def remove_empty_directories(path: Union[str, os.PathLike]) -> None:

    """Removes all empty directories under the given path.



    The function recursively checks subdirectories, following symlinks if they are present and

    traversing into them as well.



    Args:

        path: The directory path to start the removal process.

    """

    if os.path.isdir(path):

        for entry in os.listdir(path):

            entry_path = os.path.join(path, entry)

            remove_empty_directories(entry_path)

            if os.path.isdir(entry_path) and not os.listdir(entry_path):

                os.rmdir(entry_path)

