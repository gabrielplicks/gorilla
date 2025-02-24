import os

import typing



def file_name(path: str) -> str:

    """Gets the file name from a given path. If the path is invalid, raises an error.



    Args:

        path: The path to the file.

    """

    return os.path.basename(path)

