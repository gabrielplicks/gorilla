import os

import re



def scan_files(directory: str) -> list:

    """

    Scans a directory and its subdirectories for files that match a specific pattern.

    Args:

        directory: The directory to scan.

    Returns:

        A list of file paths that match the pattern.

    """

    pattern = re.compile(r'index.*\.rst')

    files = []

    for root, _, filenames in os.walk(directory):

        for filename in filenames:

            if pattern.search(filename):

                files.append(os.path.join(root, filename))

    return files

