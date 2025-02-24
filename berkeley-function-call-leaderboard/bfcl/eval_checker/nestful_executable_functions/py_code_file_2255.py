from typing import TypeVar



T = TypeVar("T", int, float)



def integer_div(dividend: T, divisor: T) -> T:

    """

    Performs integer division of two numbers.



    Args:

        dividend: The number to be divided.

        divisor: The number to divide by.



    Returns:

        The quotient of the integer division.



    Raises:

        ZeroDivisionError: If the divisor is 0.

    """

    if divisor == 0:

        raise ZeroDivisionError("Cannot divide by zero")

    return dividend // divisor

