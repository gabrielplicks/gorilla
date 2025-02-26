import re

from typing import AnyStr



def check_format(string: AnyStr) -> bool:

    """Checks if a string is in the format "<type>-<name>", where <type> is either "f" or "r", and <name> is an alphanumeric word.



    Args:

        string: The string to check.



    Returns:

        True if the string is in the correct format, False otherwise.

    """

    pattern = r"^[fr]-[a-zA-Z0-9]+$"

    return bool(re.match(pattern, string))

