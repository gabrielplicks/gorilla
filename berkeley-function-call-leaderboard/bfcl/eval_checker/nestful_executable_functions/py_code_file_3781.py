from typing import Union



def divide(a: Union[int, float], b: Union[int, float]) -> float:

    """Divides `a` by `b` and returns the quotient as a float.



    If `b` is zero, raises a `ZeroDivisionError` exception.



    Args:

        a: The numerator.

        b: The denominator.



    Raises:

        ZeroDivisionError: If `b` is zero.

    """

    try:

        return a / b

    except ZeroDivisionError:

        raise ZeroDivisionError("Division by zero is not allowed")

