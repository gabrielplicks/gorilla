from typing import Tuple



def binary_string_to_integer(binary_string: str) -> int:

    """Converts a binary string to an integer.



    Args:

        binary_string: A string of 1s and 0s representing a binary number.



    Returns:

        The integer representation of the binary string.

    """

    binary_string = binary_string[::-1]  # Reverse the string

    result = 0

    for index, bit in enumerate(binary_string):

        if bit == '1':

            result += 2 ** index  # Add 2 to the power of the index if bit is '1'

    return result

