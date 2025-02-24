from typing import Union



def power_without_pow(base: Union[int, float], exponent: int) -> Union[int, float]:

    """Calculates the power of a number, given its base and exponent, without using the `pow` built-in function or the ** operator.



    Uses bit shifting to reduce time complexity from O(n) to O(log n).



    Args:

        base: The base of the power.

        exponent: The exponent of the power.

    """

    result = 1

    while exponent > 0:

        if exponent % 2 == 1:

            result *= base

        base *= base

        exponent //= 2

    return result

