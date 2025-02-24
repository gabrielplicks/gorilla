from typing import Tuple



def floor_divide_without_div(n: int, d: int) -> int:

    """Computes the floor of `n / d` without using division, floating-point operations, built-in functions like `math.floor`, or converting it into computing the ceiling. Instead, use integer division to achieve the result.



    Args:

        n: The numerator.

        d: The denominator.

    """

    quotient = n // d  # Integer division

    if quotient * d > n:  # Check if the quotient is less than the actual result

        return quotient - 1  # Adjust the quotient to fit the floor range

    return quotient

