import re

import typing



def is_alphanumeric_underscore(s: str) -> bool:

    """Checks if a given string contains only alphanumeric characters and underscores.



    Args:

        s: The input string.



    Returns:

        True if the string is valid, False otherwise.

    """

    return bool(re.match(r'^[a-zA-Z0-9_]+$', s))

