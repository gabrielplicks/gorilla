import os

from typing import List



def convert_paths_to_basenames(file_paths: List[str]) -> List[str]:

    """Converts a list of file paths to a list of basenames (i.e., the file name without the directory path),

    which are extracted from the corresponding file paths. The function also removes any leading or trailing

    spaces from the basenames.



    Args:

        file_paths: A list of file paths.



    Returns:

        A list of basenames.

    """

    basenames = []

    for path in file_paths:

        basename = os.path.basename(path)

        basename = basename.strip()

        basenames.append(basename)

    return basenames

