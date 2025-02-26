from typing import Union



def convert_to_string_literal(input_string: str) -> str:

    """Converts a string to its equivalent string literal, with single quotes replaced by double quotes, and vice versa.



    Args:

        input_string: The input string to be converted.



    Returns:

        The resulting string literal.

    """

    string_literal = ""

    for char in input_string:

        if char == "'":

            string_literal += '"'

        elif char == '"':

            string_literal += "'"

        else:

            string_literal += char

    return string_literal

