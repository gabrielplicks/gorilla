from typing import Union



def decimal_integer_to_binary_string(n: Union[int, float]) -> str:

    """Converts a decimal integer to a binary string using the standard "string of 1's and 0's" format.

    Returns an empty string if the input is not a non-negative integer.



    Args:

        n: The decimal integer to convert to a binary string.

    """

    if isinstance(n, int) and n >= 0:

        return bin(n)[2:]  # Strip the first two characters (0b)

    else:

        return ""

