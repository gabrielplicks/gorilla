from typing import Union



def sum_of_cubes(n: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of the cubes of the first n integers.

    Args:

        n: The number of integers to sum the cubes of.

    """

    if n == 1:

        return 1

    else:

        return n ** 3 + sum_of_cubes(n - 1)



def sum_of_cubes(n: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of the cubes of the first n integers.

    Args:

        n: The number of integers to sum the cubes of.

    """

    sum_ = 0

    for i in range(1, n + 1):

        sum_ += i ** 3

    return sum_

