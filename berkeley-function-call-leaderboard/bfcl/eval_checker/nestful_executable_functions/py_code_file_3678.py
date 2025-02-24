from typing import Union



def generate_integer_string(integer: int) -> str:

    """Generates a string representation of a given integer.

    Args:

        integer: The integer to be represented as a string.

    """

    if integer == 0:

        return "0"

    elif integer < 0:

        return "-" + str(abs(integer))

    else:

        return "+" + str(integer)

