import os

from typing import List



def list_file_paths(directory: str, pattern: str) -> List[str]:

    """Lists the file paths in a directory, recursively including files within subdirectories,

    matching a given pattern, and returning the full path of each matching file.



    Args:

        directory: The directory path.

        pattern: The file pattern to match.



    Returns:

        A list of file paths.

    """

    file_paths = []



    for root, _, files in os.walk(directory):

        for file in files:

            if file.endswith(pattern):

                file_path = os.path.join(root, file)

                file_paths.append(file_path)



    return file_paths

