import re

import typing



def find_first_enclosed_substring(input_string: str) -> typing.Optional[str]:

    """Finds the first substring enclosed by parentheses or braces in the input string.



    Args:

        input_string: The input string to search for enclosed substrings.



    Returns:

        The first substring enclosed by parentheses or braces, or None if no such substring is found.

    """

    match = re.search(r'\((.*?)\)|\{(.*?)\}', input_string)

    if match:

        return match.group(1) or match.group(2)

    else:

        return None

