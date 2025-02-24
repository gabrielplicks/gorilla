from typing import Union



def almost_equal(number1: Union[int, float], number2: Union[int, float], tolerance: float) -> bool:

    """Compares two numbers for almost equality, taking a tolerance parameter for the comparison.

    Args:

        number1: The first number to compare.

        number2: The second number to compare.

        tolerance: The maximum difference between the two numbers for them to be considered almost equal.

    """

    return abs(number1 - number2) <= tolerance

