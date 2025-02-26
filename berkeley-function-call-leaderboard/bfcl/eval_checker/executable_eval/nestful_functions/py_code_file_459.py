from typing import Optional



def pow_decimal(x: float, y: float, precision: Optional[int] = 10) -> float:

    """

    Computes `x**y` with the specified precision. The default precision is 10.

    Args:

        x: The base number.

        y: The exponent.

        precision: The number of decimal places to round the result to. Default is 10.

    """

    value = x ** y

    rounded = round(value, precision)

    return rounded

