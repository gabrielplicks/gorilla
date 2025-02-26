from typing import Tuple



def exp_n(x: float, n: int) -> float:

    """Calculates the value of x raised to the power of n using a fast exponentiation algorithm.



    Args:

        x: The base number.

        n: The exponent.



    Returns:

        The value of x raised to the power of n.

    """

    y = x

    z = 1

    while n > 0:

        if n % 2 == 1:

            z *= y

        n //= 2

        y *= y

    return z

