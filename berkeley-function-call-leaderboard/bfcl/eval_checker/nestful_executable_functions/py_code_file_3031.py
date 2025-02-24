from typing import Tuple



def text_file_to_string(path: str) -> str:

    """Extracts the content of a text file as a string.

    Args:

        path: The path to the text file.

    Returns:

        The content of the file as a string. If the file does not exist, an empty string is returned.

    Raises:

        FileNotFoundError: If the file does not exist.

    """

    try:

        with open(path, 'r') as file:

            content = file.read()

    except FileNotFoundError:

        content = ''

    return content

