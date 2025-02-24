from typing import AnyStr



def to_lower(string: AnyStr) -> AnyStr:

    """Converts a string to lower case, preserving whitespace and punctuation.



    Args:

        string: The input string.



    Returns:

        The input string in lower case.

    """

    output = ''

    for char in string:

        if char.isalpha():

            output += char.lower()

        else:

            output += char

    return output

