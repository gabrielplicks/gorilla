import os

import typing



def make_absolute(path: str, is_absolute: bool) -> str:

    """

    Converts a path to an absolute path if it is not already absolute.



    Args:

        path: The path to convert.

        is_absolute: A boolean indicating whether the path is already absolute.

    """

    if is_absolute:

        return path

    else:

        current_working_directory = os.getcwd()

        return os.path.join(current_working_directory, path)

