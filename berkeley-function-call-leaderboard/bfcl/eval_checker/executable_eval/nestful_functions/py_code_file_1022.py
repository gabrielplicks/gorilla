import os

from typing import Tuple



def find_file_by_pattern(directory_path: str, pattern: str) -> Tuple[str, str]:

    """Searches for files in the directory that match the pattern.

    If a match is found, returns a tuple of the file path and the file name.

    If multiple matches are found, returns only the first match.

    If no match is found, returns None.



    Args:

        directory_path: The path of the directory to search in.

        pattern: The pattern to search for in the file names.

    """

    if not os.path.exists(directory_path):

        raise FileNotFoundError(f"Directory {directory_path} not found.")

    for root, _, files in os.walk(directory_path):

        for file in files:

            if file.startswith(pattern):

                return (os.path.join(root, file), file)

    return None

