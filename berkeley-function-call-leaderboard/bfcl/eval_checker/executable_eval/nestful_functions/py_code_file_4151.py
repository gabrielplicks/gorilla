import re

import typing



def count_uppercase_letters(string: str) -> typing.Optional[int]:

    """Counts the number of uppercase letters within a given string.



    Args:

        string: The input string.



    Returns:

        The number of uppercase letters in the string, or `None` if the string is empty or consists only of whitespace characters.

    """

    if string.isspace():

        return None

    else:

        uppercase_letters = re.findall(r'[A-Z]', string)

        return len(uppercase_letters) or None

