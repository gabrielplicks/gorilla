from typing import Union



def format_file_path(path: Union[str, bytes]) -> str:

    """Formats a file path.



    If the file path is a relative path, the function returns the path starting with './',

    otherwise, the function returns the path without any changes.



    Args:

        path: The file path to format.

    """

    if path[0] != '/':

        return f'./{path}'

    return path

