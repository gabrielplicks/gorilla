from typing import List, Union



def format_directory(directory_path: str) -> str:

    """Formats a directory path into a string with each directory or file on a new line.



    Args:

        directory_path: The directory path to format.



    Returns:

        A string with each directory or file on a new line.

    """

    directory_path_parts: List[str] = directory_path.split('/')

    formatted_directory: str = ''



    for part in directory_path_parts:

        formatted_directory += part + '/' + '\n'



    return formatted_directory

