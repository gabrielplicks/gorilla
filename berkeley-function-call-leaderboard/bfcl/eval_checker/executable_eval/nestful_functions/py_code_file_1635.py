from typing import Tuple



def divide_without_div_operator(dividend: int, divisor: int) -> Tuple[int, int]:

    """Divides two integers without using the `//` operator, and returns the quotient and remainder.



    Args:

        dividend: The number to be divided.

        divisor: The number to divide by.



    Raises:

        ZeroDivisionError: If the divisor is zero.

    """

    quotient = 0

    remainder = dividend

    if divisor == 0:

        raise ZeroDivisionError("Cannot divide by zero")

    while remainder >= divisor:

        remainder -= divisor

        quotient += 1

    return quotient, remainder

