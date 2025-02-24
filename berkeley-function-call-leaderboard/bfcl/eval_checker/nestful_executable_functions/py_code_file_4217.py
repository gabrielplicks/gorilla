from typing import Union



def divide_zero_safe(num: Union[int, float], denom: Union[int, float]) -> Union[int, float]:

    """

    Performs a division of two numbers in a zero-safe manner.

    If `denom == 0`, the function returns `float('inf')` if `num > 0`,

    `-float('inf')` if `num < 0`, and 0 if `num == 0`.



    Args:

        num: The numerator.

        denom: The denominator.

    """

    if denom == 0:

        if num > 0:

            return float('inf')

        elif num < 0:

            return -float('inf')

        else:

            return 0

    return num / denom

