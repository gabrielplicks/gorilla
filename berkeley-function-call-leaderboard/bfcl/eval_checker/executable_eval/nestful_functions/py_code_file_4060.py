from typing import Union



def check_pentagonal(n: int) -> bool:

    """

    Returns True if the given number n is a pentagonal number and False otherwise.

    Args:

        n: The number to check if it is a pentagonal number.

    """

    root1 = (1 + (1 + 24 * n) ** 0.5) / 6

    root2 = (1 - (1 + 24 * n) ** 0.5) / 6

    return int(root1) == root1 or int(root2) == root2

