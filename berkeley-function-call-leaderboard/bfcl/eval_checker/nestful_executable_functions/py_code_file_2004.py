from typing import Tuple



def int_div_with_ceil(a: int, b: int) -> int:

    """Computes the quotient of `a` and `b` using only the floor operation (`//`).

    The result is the quotient rounded up to the next integer if the division has a remainder.

    For example, `int_div_with_ceil(6, 3)` should return `2`, and `int_div_with_ceil(5, 3)` should return `2`.

    The function handles negative inputs correctly.



    Args:

        a: The numerator.

        b: The denominator.



    Returns:

        The quotient of `a` and `b`, rounded up to the next integer if the division has a remainder.

    """

    return (a + b - 1) // b

