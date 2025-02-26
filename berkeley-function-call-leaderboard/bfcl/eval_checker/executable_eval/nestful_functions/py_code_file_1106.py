from typing import Tuple



def egcd(a: int, b: int) -> Tuple[int, int, int]:

    """Calculates the greatest common divisor (gcd) of two numbers and returns a tuple (gcd, x, y) such that ax + by = gcd.



    Args:

        a: The first number.

        b: The second number.



    Returns:

        A tuple (gcd, x, y) such that ax + by = gcd.

    """

    if b == 0:

        return (a, 1, 0)

    gcd, x, y = egcd(b, a % b)

    return (gcd, y, x - (a // b) * y)

