from typing import Union



def is_pow_two(n: Union[int, float]) -> bool:

    """Determines if a given positive integer `n` is a power of 2.



    Args:

        n: A positive integer.



    Returns:

        True if `n` is a power of 2, and False otherwise.

    """

    k = 0

    while k <= n:

        if 2 ** k == n:

            return True

        k += 1

    return False

