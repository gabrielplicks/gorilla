import os

from typing import List



def extract_file_or_folder_names(path: str) -> List[str]:

    """Extracts the file or folder names from a file path.



    Args:

        path: The file path.



    Returns:

        A list of strings that are each a file or folder name in the path.

    """

    directory, filename = os.path.split(path)

    if directory == os.path.sep:

        return [filename]

    directory_parts = os.path.split(directory)

    if directory_parts[1]:

        directory_parts[1].append(directory_parts[1])

    return directory_parts + [filename]

