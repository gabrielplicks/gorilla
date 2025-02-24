from typing import Tuple



def both_even(n1: int, n2: int) -> bool:

    """Checks if both input integers are even using bitwise operations.



    Args:

        n1: The first integer.

        n2: The second integer.



    Returns:

        A boolean value indicating whether both numbers are even.

    """

    return (n1 & 1) == 0 and (n2 & 1) == 0

