from typing import Tuple



def calculate_sum_and_percentage(a: int, b: int, c: int, d: int) -> Tuple[int, int]:

    """Calculates the sum of a and b divided by c rounded to the nearest integer, and the percentage of d out of c rounded to the nearest integer.



    Args:

        a: The first number.

        b: The second number.

        c: The divisor.

        d: The number to calculate the percentage of.



    Returns:

        A tuple of two integers.

    """

    e = round((a + b) / c)

    f = round((d / c) * 100)

    return (e, f)

