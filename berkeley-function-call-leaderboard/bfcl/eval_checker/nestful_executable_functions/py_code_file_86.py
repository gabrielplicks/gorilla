import os

from pathlib import Path

from typing import List



def find_files_by_extension(directory_path: str, extension: str) -> List[Path]:

    """Finds all files within the directory tree that match the given file extension.



    Args:

        directory_path: The path to the directory to search.

        extension: The file extension to match.



    Returns:

        A list of file paths that match the given extension.

    """

    matched_files = []

    for root, directories, files in os.walk(directory_path):

        for file in files:

            file_path = Path(root).joinpath(file)

            if file_path.suffix == extension:

                matched_files.append(file_path)

    return matched_files

