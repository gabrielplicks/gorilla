import re

from typing import Set



def special_chars_to_underscore(string: str) -> str:

    """Replaces all special non-alphanumeric characters in a string with underscores.



    Args:

        string: The input string.



    Returns:

        A new string with all special characters replaced with underscores.

    """

    pattern = re.compile('[^A-Za-z0-9-]')

    special_chars: Set[str] = set(pattern.findall(string))



    for char in special_chars:

        string = string.replace(char, '_')



    return string

