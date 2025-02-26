from typing import Union



def fast_exponentiation(base: Union[int, float], power: int) -> Union[int, float]:

    """Calculates an integer or float raised to a non-negative integer power.



    Args:

        base: The base of the exponentiation.

        power: The power to raise the base to.



    Raises:

        ValueError: If the power is negative.

    """

    if power < 0:

        raise ValueError("Power must be non-negative")



    result = 1

    while power > 0:

        if power % 2 == 1:

            result *= base

        base *= base

        power //= 2



    return result

