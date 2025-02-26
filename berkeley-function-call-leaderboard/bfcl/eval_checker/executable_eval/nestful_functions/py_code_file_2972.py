from math import ceil

from typing import Union



def round_to_ceiling(number: Union[int, float]) -> int:

    """Rounds a number to the nearest whole number, but always up.



    Args:

        number: The number to round.



    Returns:

        The nearest whole number that is greater than or equal to the given number.

    """

    return ceil(number)

