from typing import Union



def divide_int(dividend: int, divisor: int) -> int:

    """Calculates the quotient of two integers, rounded down to the nearest integer.

    If the denominator is 0, returns 0.

    Args:

        dividend: The numerator.

        divisor: The denominator.

    """

    try:

        return dividend // divisor

    except ZeroDivisionError:

        return 0

