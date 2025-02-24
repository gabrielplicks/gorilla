from typing import Tuple



def calculate_hexadecimal_addition(hex1: str, hex2: str) -> str:

    """Calculates the sum of two hexadecimal numbers.



    Args:

        hex1: The first hexadecimal number as a string.

        hex2: The second hexadecimal number as a string.



    Returns:

        The sum of the two hexadecimal numbers as a hexadecimal string.

    """

    int1 = int(hex1, 16)

    int2 = int(hex2, 16)

    sum = int1 + int2

    return hex(sum)

