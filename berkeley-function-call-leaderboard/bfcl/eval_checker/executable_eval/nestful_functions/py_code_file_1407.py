from typing import Tuple



def calculate_gcd(a: int, b: int) -> int:

    """Calculates the greatest common divisor of two positive integers.



    Args:

        a: The first positive integer.

        b: The second positive integer.



    Returns:

        The greatest common divisor of a and b.

    """

    if b > a:

        a, b = b, a

    remainder = a % b

    while remainder != 0:

        a = b

        b = remainder

        remainder = a % b

    return b

