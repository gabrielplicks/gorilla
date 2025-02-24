from typing import Union



def is_power_of_three(n: Union[int, float]) -> bool:

    """Determines if a given positive integer `n` is a power of three.



    Args:

        n: The positive integer to check.



    Returns:

        True if `n` is a power of three, False otherwise.

    """

    if n <= 0:

        return False

    while n % 3 == 0:

        n //= 3

    return n == 1

