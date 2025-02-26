from typing import AnyStr



def normalize_case(input_string: AnyStr) -> AnyStr:

    """Normalizes a string by converting all upper-case characters to lower-case and all lower-case characters to upper-case.



    Args:

        input_string: The input string to be normalized.



    Returns:

        The normalized string.

    """

    normalized_string = ''

    for char in input_string:

        if char.isalpha():

            normalized_string += char.swapcase()

        else:

            normalized_string += char

    return normalized_string

