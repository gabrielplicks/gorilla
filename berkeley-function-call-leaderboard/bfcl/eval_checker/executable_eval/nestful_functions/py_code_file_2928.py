from typing import Union



def bit_repr(num: Union[int, float]) -> str:

    """Calculates the binary representation of a positive integer.

    Args:

        num: The positive integer to be converted to binary.

    """

    binary_repr = ''

    while num > 0:

        binary_repr += str(num & 1)  # Extract the LSB using bitwise AND

        num >>= 1  # Shift the integer to the right by one bit

    return binary_repr[::-1]  # Reverse the binary representation to get the correct order

