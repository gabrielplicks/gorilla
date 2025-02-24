from math import factorial

from typing import Union



def permutation(n: int, k: int) -> Union[int, float]:

    """Calculates the number of permutations of `k` items chosen from a set of `n` items.

    Args:

        n: The total number of items.

        k: The number of items to choose.

    Returns:

        The number of permutations if `k` is less than or equal to `n`, otherwise `-1`.

    """

    if k > n:

        return -1

    return factorial(n) // factorial(n - k)

