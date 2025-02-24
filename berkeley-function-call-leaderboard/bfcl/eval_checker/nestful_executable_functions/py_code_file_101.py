import os

import fnmatch

from typing import List



def get_file_names(dir: str, ext: str = None) -> List[str]:

    """

    Returns a list of file names in the directory that have the given extension.

    Args:

        dir: The directory to search in.

        ext: The file extension to search for. If not given, all files are returned.

    Raises:

        IOError: If an IO error occurs.

    """

    try:

        files = []

        for file in os.listdir(dir):

            if ext:

                if fnmatch.fnmatch(file, f'*.{ext}'):

                    files.append(file)

            else:

                files.append(file)

        return files

    except IOError as e:

        raise e

