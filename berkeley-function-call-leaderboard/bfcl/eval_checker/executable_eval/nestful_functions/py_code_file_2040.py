import os

from typing import Union



def create_or_return_directory(directory: str) -> Union[str, None]:

    """Creates a directory if it does not exist, and returns the directory.

    If the directory already exists, the function simply returns the directory.

    Args:

        directory: The directory to create or return.

    """

    if os.path.exists(directory):

        return directory

    else:

        os.makedirs(directory)

        return directory

