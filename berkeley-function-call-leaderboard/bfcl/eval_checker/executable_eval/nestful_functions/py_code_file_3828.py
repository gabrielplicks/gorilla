from typing import Union



def reformat_path(path: str) -> str:

    """Reformats a path string to ensure it follows a specific format.



    Args:

        path: The input path string.



    Returns:

        The reformatted path string.

    """

    path = path.replace('\\', '/')

    path = path.strip('/')

    path = path.replace('//', '/')

    if path != '/':

        path = path.rstrip('/')



    return path

