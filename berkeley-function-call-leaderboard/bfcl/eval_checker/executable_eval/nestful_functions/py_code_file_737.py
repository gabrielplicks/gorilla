import re

from typing import Union



def contains_only_alphabetic(string: Union[str, bytes]) -> bool:

    """Checks if a given string contains only alphabetic characters using regular expressions.

    Args:

        string: The string to check.

    """

    pattern = re.compile(r"^[a-zA-Z]+$")

    return bool(pattern.search(string))

