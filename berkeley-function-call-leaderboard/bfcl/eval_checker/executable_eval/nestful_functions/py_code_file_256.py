from typing import Any



def modular_exponentiation(a: int, b: int, c: int) -> int:

    """Calculates the remainder of a number `a` raised to a power `b` modulo `c`.

    Args:

        a: The base number.

        b: The exponent.

        c: The modulus.

    """

    result = 1

    for _ in range(b):

        result = (result * a) % c

    return result

