from typing import Union



def is_square_root(n: int, m: int) -> bool:

    """Checks if m is a perfect square of n.



    Args:

        n: A positive integer.

        m: A positive integer.



    Returns:

        A boolean value indicating whether m is a perfect square of n.

    """

    if m % n != 0:

        return False



    quotient = m // n

    return is_square_root_helper(quotient)



def is_square_root_helper(n: int) -> bool:

    """Checks if a number is a perfect square.



    Args:

        n: A positive integer.



    Returns:

        A boolean value indicating whether n is a perfect square.

    """

    if n == 1:

        return True



    for i in range(1, n // 2 + 1):

        if i * i == n:

            return True



    return False

