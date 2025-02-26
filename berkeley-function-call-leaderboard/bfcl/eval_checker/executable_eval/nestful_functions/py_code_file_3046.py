from typing import Union



def format_with_leading_zeros(integer: Union[int, float], width: int) -> str:

    """Formats a given integer into a string with a fixed width and leading zeros.

    Args:

        integer: The integer to format.

        width: The desired width of the formatted string.

    """

    return '{:0>{width}}'.format(integer, width=width)

