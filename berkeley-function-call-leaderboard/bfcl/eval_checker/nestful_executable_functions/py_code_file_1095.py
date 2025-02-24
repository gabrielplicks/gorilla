from typing import List



def factor(n: int) -> List[int]:

    """Calculates the factors of a given integer `n` and returns them in a list.



    Args:

        n: The integer to calculate the factors of.



    Returns:

        A list of factors of the given integer.

    """

    factors = []

    for i in range(1, n + 1):

        if n % i == 0:

            factors.append(i)

    return factors

