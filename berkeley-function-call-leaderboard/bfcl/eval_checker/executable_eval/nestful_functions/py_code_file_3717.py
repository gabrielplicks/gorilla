from typing import Union



def unsigned_int_to_binary(n: Union[int, float]) -> str:

    """Converts a 32-bit unsigned integer into its binary representation (a string of 0s and 1s) without using any built-in functions.



    Args:

        n: The 32-bit unsigned integer to be converted.



    Returns:

        The binary representation of the input integer.

    """

    assert isinstance(n, int) and n >= 0, "n must be a non-negative integer"

    assert n <= 2**32 - 1, "n must be a 32-bit unsigned integer"

    binary_repr = ""

    for i in range(32):

        binary_repr += "1" if (n & 1) else "0"

        n >>= 1



    return binary_repr[::-1]  # Reverse the bits

