from typing import List

import os



def get_file_extensions(file_paths: List[str]) -> List[str]:

    """Extracts file extensions from a list of file paths.



    Args:

        file_paths: A list of file paths.



    Returns:

        A list of file extensions, excluding the dot ('.') character.

    """

    file_extensions = [os.path.splitext(path)[1][1:] for path in file_paths]

    return file_extensions

