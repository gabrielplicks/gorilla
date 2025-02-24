from typing import Tuple



def multiply_integers(a: int, b: int) -> int:

    """Multiplies two integers `a` and `b` and returns their product.



    Args:

        a: The first integer.

        b: The second integer.



    Returns:

        The product of `a` and `b`.

    """

    result = 0

    for _ in range(abs(b)):

        result += a

    if b < 0:

        result = -result



    return result

