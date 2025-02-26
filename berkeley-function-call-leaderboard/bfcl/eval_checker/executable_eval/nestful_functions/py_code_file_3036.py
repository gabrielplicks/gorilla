from typing import Union



def int_to_bitwise_string(num: Union[int, float]) -> str:

    """Converts a positive integer to a bitwise string.

    The function accepts an integer and returns a string containing the binary representation of the integer,

    where the least significant bit is at the beginning of the string.

    Args:

        num: The positive integer to be converted.

    """

    result = ''

    while num > 0:

        result = str(num % 2) + result

        num //= 2

    return result

