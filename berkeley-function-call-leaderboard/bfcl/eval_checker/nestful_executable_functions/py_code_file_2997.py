import os

from typing import List



def get_directories_from_path(path: str) -> List[str]:

    """

    Returns a list of strings representing the individual directories in the given path.

    The function handles both forward slash (/) and back slash (\) as directory separators,

    and returns the correct path strings based on the underlying operating system.



    Args:

        path: The path string to extract directories from.



    Returns:

        A list of strings representing the individual directories in the given path.

    """

    separator = os.sep

    if separator == '/':

        directories = path.split('/')

    else:

        directories = path.split('\\')

    directories = list(filter(None, directories))

    return directories

