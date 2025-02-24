from typing import AnyStr



def switched_case(string: AnyStr) -> AnyStr:

    """

    Replaces all lowercase letters in a string with their uppercase equivalents and vice versa.

    Non-letter characters are retained as-is.

    Args:

        string: The input string.

    """

    new_string = ''

    for char in string:

        if char.islower():

            new_string += char.upper()

        elif char.isupper():

            new_string += char.lower()

        else:

            new_string += char

    return new_string

