from typing import Any



def get_factorial(n: int) -> int:

    """Calculates the factorial of a positive integer using recursion.

    Args:

        n: The positive integer to calculate the factorial of.

    """

    if n <= 1:

        return 1

    else:

        return n * get_factorial(n - 1)

