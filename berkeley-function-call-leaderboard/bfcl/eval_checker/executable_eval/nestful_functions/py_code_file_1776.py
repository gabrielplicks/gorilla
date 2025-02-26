from typing import Union



def compose_file_path(directory_path: str, file_name: str) -> str:

    """Composes a full file path by concatenating the directory_path and file_name with a slash character.



    Args:

        directory_path: The path to the directory.

        file_name: The name of the file.



    Returns:

        The full file path.

    """

    directory_path = directory_path.rstrip('/')

    file_name = file_name.lstrip('/')

    if directory_path.endswith('/'):

        return directory_path + file_name

    return directory_path + '/' + file_name

