import re

import string



def lowercase_letters(input_str: str, default_str: str) -> str:

    """

    Checks if an input string comprises only letters and digits and has at least one letter.

    If yes, returns the string in all lowercase letters. If no, returns a default string.



    Args:

        input_str: The input string to be checked.

        default_str: The default string to be returned if the input string does not meet the criteria.

    """

    if re.match(r"^[\w\d]+$", input_str) and any(c.isalpha() for c in input_str):

        return input_str.lower()

    return default_str

