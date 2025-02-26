from typing import Tuple



def divide_without_div(a: int, b: int) -> int:

    """

    Computes `a // b` without using division or floating-point operations.



    Args:

        a: The numerator.

        b: The denominator.



    Returns:

        The integer part of the quotient.

    """

    if a < b:

        return 0



    count = 0

    while a >= b:

        a -= b

        count += 1



    return count

