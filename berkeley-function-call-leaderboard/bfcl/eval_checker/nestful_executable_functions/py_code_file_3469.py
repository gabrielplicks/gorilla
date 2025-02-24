from typing import Union



def unit_step(x: Union[int, float]) -> int:

    """Calculates the unit step function of a real number.

    Args:

        x: The real number to calculate the unit step function of.

    """

    if x <= 0:

        return 0

    else:

        return 1

