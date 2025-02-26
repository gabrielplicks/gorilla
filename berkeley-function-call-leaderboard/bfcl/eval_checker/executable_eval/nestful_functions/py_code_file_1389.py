from typing import AnyStr



def escape_string(input_string: AnyStr) -> AnyStr:

    """Escapes all special characters in a string with a backslash.



    Args:

        input_string: The input string.



    Returns:

        The input string with all special characters escaped.

    """

    new_string = ""

    for char in input_string:

        if char in "\\'\"":

            new_string += "\\"

        new_string += char

    return new_string

