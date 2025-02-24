from typing import Union



def best_round(n: Union[int, float]) -> int:

    """Calculates the smallest power of 2 greater than or equal to a given number n.



    Args:

        n: The given number.



    Returns:

        The smallest power of 2 greater than or equal to n.

    """

    result = 1

    while result < n:

        result *= 2

    return result

