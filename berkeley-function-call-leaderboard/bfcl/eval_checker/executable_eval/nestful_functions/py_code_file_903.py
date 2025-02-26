import logging

from typing import Union



def divide_by_zero(numerator: Union[int, float], denominator: Union[int, float]) -> Union[int, float]:

    """Divides the numerator by the denominator.

    If the denominator is zero, raises a ZeroDivisionError exception with the message "Cannot divide by zero."

    Args:

        numerator: The numerator of the division.

        denominator: The denominator of the division.

    """

    try:

        result = numerator / denominator

    except ZeroDivisionError:

        logging.exception("ZeroDivisionError: Cannot divide by zero.")

        raise

    else:

        return result

