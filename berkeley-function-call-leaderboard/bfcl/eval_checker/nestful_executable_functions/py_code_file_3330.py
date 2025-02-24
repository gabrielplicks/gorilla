import os

from typing import List



def find_files_with_extension(directory_path: str, extension: str) -> List[str]:

    """

    Finds all files with a specific file extension in a directory and its subdirectories.



    Args:

        directory_path: The path to the directory to search.

        extension: The file extension to search for.



    Returns:

        A list of full paths to the files with the specified extension.

    """

    files_with_extension = []

    for root, _, files in os.walk(directory_path):

        for file in files:

            if file.endswith(extension):

                full_path = os.path.join(root, file)

                files_with_extension.append(full_path)

    return files_with_extension

