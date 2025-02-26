from typing import List



def to_binary(num: int) -> str:

    """Converts a number to its binary representation.



    Args:

        num: The number to be converted.



    Returns:

        The binary representation of the number as a string.

    """

    binary_digits: List[str] = []

    while num > 0:

        binary_digits.append(str(num % 2))

        num //= 2

    return ''.join(reversed(binary_digits))

