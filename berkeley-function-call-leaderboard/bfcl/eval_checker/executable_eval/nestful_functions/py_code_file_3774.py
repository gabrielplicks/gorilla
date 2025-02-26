from typing import Tuple



def euclidean_gcd(a: int, b: int) -> int:

    """Calculates the greatest common divisor (GCD) of two integers using the Euclidean algorithm.

    Args:

        a: The larger of the two numbers.

        b: The smaller of the two numbers.

    """

    if b == 0:

        return a

    return euclidean_gcd(b, a % b)

