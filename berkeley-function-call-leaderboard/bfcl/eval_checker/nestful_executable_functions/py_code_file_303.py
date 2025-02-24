from typing import Union



def sum_multiples_of_n(n: Union[int, float], m: Union[int, float]) -> Union[int, float]:

    """Calculates the sum of all multiples of `n` that are less than or equal to `m`.



    Args:

        n: The number to find multiples of.

        m: The upper limit for the multiples.



    Returns:

        The sum of all multiples of `n` that are less than or equal to `m`.

    """

    sum = 0

    for i in range(n, m + 1, n):

        sum += i

    return sum

