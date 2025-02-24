import os

import fnmatch

from typing import List



def glob_path(path: str, pattern: str) -> List[str]:

    """Recursively searches for files matching a given pattern in a provided path and returns a list of matching file paths as strings.



    Args:

        path: The base directory to search.

        pattern: A string that can contain wildcards (e.g., `*.txt`).



    Returns:

        A list of file paths as strings.

    """

    if not os.path.isdir(path):

        return []



    matching_files = []

    for root, dirs, files in os.walk(path):

        for file in files:

            if pattern and not fnmatch.fnmatch(file, pattern):

                continue

            file_path = os.path.join(root, file)

            matching_files.append(file_path)



    return matching_files

