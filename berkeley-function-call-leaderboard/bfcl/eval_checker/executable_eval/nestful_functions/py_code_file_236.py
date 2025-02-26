from typing import Union



def max_diff(a: Union[int, float], b: Union[int, float]) -> Union[int, float]:

    """Calculates the maximum difference between two integers.



    Args:

        a: The first integer.

        b: The second integer.



    Returns:

        The maximum difference between the two integers.

    """

    if a > b:

        return a - b

    elif b > a:

        return b - a

    else:

        return 0

