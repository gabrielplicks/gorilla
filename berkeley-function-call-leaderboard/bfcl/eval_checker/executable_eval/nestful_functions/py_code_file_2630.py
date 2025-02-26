from typing import Union



def inverse_log(a: int, y: Union[int, float]) -> float:

    """Calculates the inverse of the logarithmic function f(x) = log_a(x).



    Args:

        a: The base of the logarithm. Must be a positive integer.

        y: The value to calculate the inverse of. Must be a number (either an integer or a floating-point number).



    Returns:

        The result of raising a to the power of y.

    """

    assert a > 0, "a must be a positive integer"

    assert isinstance(y, (int, float)), "y must be a number"

    return pow(a, y)

