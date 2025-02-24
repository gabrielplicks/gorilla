from typing import Union



def convert_positive_int(input: str) -> Union[int, str]:

    """

    Converts a non-negative integer string to a positive integer.

    If the input is not a non-negative integer, returns 'Not a valid integer.'.

    Args:

        input: The input string to be converted.

    """

    if input.isdigit():

        return int(input)

    else:

        return 'Not a valid integer.'

