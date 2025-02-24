from typing import Union



def sum_of_first_n_integers(n: Union[int, float]) -> int:

    """Calculates the sum of the first n integers.



    Args:

        n: A positive integer representing the number of integers to sum.



    Returns:

        The sum of the first n integers.

    """

    total = 0

    for i in range(1, int(n) + 1):

        total += i

    return total

