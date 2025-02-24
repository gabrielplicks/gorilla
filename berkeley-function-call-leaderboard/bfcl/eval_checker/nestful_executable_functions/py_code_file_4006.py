from typing import Union



def float_to_decimal(s: str) -> str:

    """Converts a string representation of a floating-point number to a decimal fraction.



    Args:

        s: The input string representing a floating-point number.



    Returns:

        A string representing the same number, but in decimal fraction format.

        If the fractional part is zero, it is removed.

    """

    x = float(s)

    return f'{x:.6f}'.rstrip('0').rstrip('.')

