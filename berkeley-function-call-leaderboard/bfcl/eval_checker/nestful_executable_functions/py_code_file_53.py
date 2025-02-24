import os

import fnmatch

from typing import List



def find_log_files(directory: str) -> List[str]:

    """Finds all the log files in the given directory and its subdirectories.

    Args:

        directory: The directory to search for log files.

    Returns:

        A list of all the file paths that correspond to the files that match the pattern `*.log`.

    """

    log_files = []

    for root, dirs, files in os.walk(directory):

        for filename in fnmatch.filter(files, '*.log'):

            log_file_path = os.path.join(root, filename)

            log_files.append(log_file_path)

    return log_files

