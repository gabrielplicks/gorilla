from typing import Union



def pow_2_test(n: Union[int, float]) -> bool:

    """Determines whether the given number `n` is a power of 2.



    Args:

        n: The number to check.



    Raises:

        ValueError: If `n` is not an integer.

    """

    if not isinstance(n, int):

        raise ValueError('n must be an integer')

    return n > 0 and n & (n - 1) == 0

