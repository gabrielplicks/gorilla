from typing import Tuple



def int_div_ceil(a: int, b: int) -> int:

    """Calculates the integer division of `a` and `b` rounded up.



    Args:

        a: The numerator.

        b: The denominator.



    Returns:

        The integer division of `a` and `b` rounded up.

    """

    quotient, remainder = divmod(a, b)

    return quotient + (1 if remainder else 0)

