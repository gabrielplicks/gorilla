from typing import Tuple



def calculate_mod(a: int, b: int) -> int:

    """Calculates the modulus of the given expression with a + b and returns the result.



    Args:

        a: A positive integer.

        b: A positive integer.



    Returns:

        The modulus of the given expression with a + b.

    """

    return (a * b + b // 2) % (a + b)

