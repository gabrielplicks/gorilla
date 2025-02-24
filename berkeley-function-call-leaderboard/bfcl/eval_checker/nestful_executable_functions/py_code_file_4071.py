import re

from typing import Tuple



def replace_non_surrounded_substring(string: str, old: str, new: str) -> str:

    """Replaces all occurrences of a given substring in a string with another substring, but only if the substring is not surrounded by spaces.



    Args:

        string: The input string.

        old: The substring to be replaced.

        new: The substring to replace with.



    Returns:

        The modified string.

    """

    matches = re.finditer(r'(?<!\S)' + re.escape(old) + r'(?!\S)', string)

    for match in matches:

        string = string.replace(match.group(0), new)



    return string

