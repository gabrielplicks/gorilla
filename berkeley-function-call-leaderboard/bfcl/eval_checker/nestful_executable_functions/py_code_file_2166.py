import pathlib

import re

import os



def find_files_by_content(directory: str, string: str) -> list:

    """Finds files in a directory and its subdirectories that contain a specific string within their contents.



    Args:

        directory: The directory to search in.

        string: The string to search for within the files' contents.



    Returns:

        A list of file paths that contain the specified string within their contents.

    """

    file_paths = []



    for path in pathlib.Path(directory).rglob('*'):

        if path.is_file():

            try:

                with path.open('r') as file:

                    content = file.read()

                    match = re.search(string, content)

                    if match:

                        file_paths.append(str(path))

            except OSError:

                pass



    return file_paths

