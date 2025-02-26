from typing import Union



def reversed_digits(x: Union[int, str]) -> str:

    """

    Returns a string representing the digits of the input integer `x` in reverse order.

    The digits are separated by commas (`,`).



    Args:

        x: The input integer.

    """

    digits = str(x)[::-1]  # Split the integer into its digits and reverse the order

    return ", ".join(digits)  # Join the digits with commas (,)

