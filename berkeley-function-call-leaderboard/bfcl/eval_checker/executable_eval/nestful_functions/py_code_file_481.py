from typing import Tuple



def xgcd(a: int, b: int) -> Tuple[int, int, int]:

    """Calculates the GCD (greatest common divisor) and the coefficients of the linear combination of `a` and `b` using the extended Euclidean algorithm.



    Args:

        a: The first integer.

        b: The second integer.



    Returns:

        A tuple of three elements, `(gcd, x, y)`, where `gcd` is the GCD of `a` and `b`, and `x` and `y` are the coefficients such that `ax + by = gcd`.

    """

    if b == 0:

        return a, 1, 0

    else:

        gcd, x, y = xgcd(b, a % b)

        return gcd, y, x - (a // b) * y

