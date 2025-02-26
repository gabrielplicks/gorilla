from typing import Union



def bit_representation(n: Union[int, float]) -> str:

    """Returns the 32-bit representation of a 32-bit unsigned integer `n` as a string.

    The least significant bit is the first character and the most significant bit is the last character.

    Args:

        n: A 32-bit unsigned integer.

    """

    binary_string = ''

    while n > 0:

        binary_string = str(n % 2) + binary_string

        n //= 2

    binary_string = binary_string.zfill(32)

    return binary_string

