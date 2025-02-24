import string

from typing import Union



def convert_to_url_path(path: Union[str, None]) -> str:

    """Converts a string path to a URL-safe path.

    Removes all non-alphanumeric characters from the path, replaces spaces with underscores,

    and converts the path to lowercase. Handles any potential errors gracefully.

    Args:

        path: The string path to convert.

    Returns:

        The URL-safe path.

    """

    if not isinstance(path, str) or not path:

        return ""

    url_path = ""

    for char in path:

        if char.isalnum() or char == " ":

            url_path += char

    url_path = url_path.replace(" ", "_").lower()



    return url_path

