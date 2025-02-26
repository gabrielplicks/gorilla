from typing import Tuple



def gcd_of_modules(a: int, b: int) -> int:

    """Calculates the greatest common divisor of `a mod b` and `b mod a`.



    Args:

        a: A positive integer.

        b: A positive integer.



    Returns:

        The greatest common divisor of `a mod b` and `b mod a`.

    """

    if a < b:

        a, b = b, a  # Ensure a >= b



    while b > 0:

        a, b = b, a % b  # Divide the larger number by the smaller one



    return a

