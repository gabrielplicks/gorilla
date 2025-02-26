from typing import Union



def absolute_sum(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:

    """Calculates the absolute sum of two numbers based on their signs.

    Args:

        a: The first number.

        b: The second number.

    """

    if a > 0 and b > 0:

        return abs(a + b)

    elif a < 0 and b < 0:

        return -abs(a + b)

    else:

        return 0

