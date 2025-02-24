from typing import Union



def smallest_x(n: Union[int, float]) -> Union[int, float]:

    """Returns the smallest x such that x * x > n for a given n.



    Args:

        n: The given number.



    Returns:

        The smallest x such that x * x > n.

    """

    low = 1

    high = n

    while low <= high:

        mid = (low + high) // 2

        if mid * mid <= n:

            low = mid + 1

        else:

            high = mid - 1

    return low

