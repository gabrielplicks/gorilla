from typing import AnyStr



def validate_ascii_and_whitespace(input_string: AnyStr) -> str:

    """

    Returns a new string that only contains valid ASCII characters and whitespace.

    The function removes all non-ASCII characters, and also removes any control characters such as '\t', '\n', and '\r' from the input string.

    Args:

        input_string: The input string to be validated.

    """

    output_string = ''

    for character in input_string:

        code_point = ord(character)

        if code_point < 128 and code_point not in [9, 10, 13]:

            output_string += character

    return output_string

