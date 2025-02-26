import re

import typing



def remove_consecutive_spaces(s: str) -> str:

    """Removes all consecutive spaces in a string using regular expressions.



    Args:

        s: The input string.



    Returns:

        The cleaned string with all consecutive spaces replaced by a single space.

    """

    return re.sub(r'\s+', ' ', s)

