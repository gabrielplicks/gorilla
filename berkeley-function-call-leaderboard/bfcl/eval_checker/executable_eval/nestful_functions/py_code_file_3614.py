from typing import Tuple



def divide_integer(a: int, b: int) -> int:

    """Calculates the integer quotient of two integers `a` and `b`.



    Args:

        a: The dividend.

        b: The divisor.



    Returns:

        The integer quotient of `a` and `b`.

    """

    quotient = 0

    while a >= b:

        a -= b

        quotient += 1

    return quotient

