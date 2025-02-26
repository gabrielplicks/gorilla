from typing import Union



def create_formatted_string(integer: int, padding: int) -> str:

    """Creates a formatted string based on a given integer.

    If the integer is less than 5 digits, prepend the integer with leading zeroes to make it 5 digits long.

    Otherwise, return the integer as a string.

    Args:

        integer: A positive integer.

        padding: The padding amount.

    """

    if len(str(integer)) < 5:

        return f'{integer:0>{padding}}'

    else:

        return str(integer)

