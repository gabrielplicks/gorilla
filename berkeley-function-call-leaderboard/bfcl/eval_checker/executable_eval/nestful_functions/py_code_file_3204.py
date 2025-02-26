from typing import Union



def sum_of_range(n: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of a range of numbers from 1 to n, inclusive.

    Args:

        n: A positive integer or float.

    """

    sum = 0

    for i in range(1, int(n) + 1):

        sum += i

    return sum

