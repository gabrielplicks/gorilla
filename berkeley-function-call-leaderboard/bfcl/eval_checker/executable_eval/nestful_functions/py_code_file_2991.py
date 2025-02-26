import os

import glob

from typing import Dict



def find_count_by_ext(directory_path: str, ext: str) -> Dict[str, int]:

    """Reads and counts the number of lines in each of the files in a given directory that match a specific file extension.



    Args:

        directory_path: The path to the directory.

        ext: The file extension to match.



    Returns:

        A dictionary that maps the file name to the line count.

    """

    if not os.path.isdir(directory_path):

        raise ValueError("Directory path is not valid.")

    files_with_ext = glob.glob(os.path.join(directory_path, f'*.{ext}'))

    line_counts = {}



    for file_path in files_with_ext:

        with open(file_path, 'r') as file:

            line_counts[file_path] = len(file.readlines())



    return line_counts

