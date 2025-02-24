from typing import Union



def recursive_sum_limit_safe(n: Union[int, float]) -> Union[int, float]:

    """Computes the sum of the first n natural numbers using recursion.

    For very large values of n, it uses the formula for the sum of consecutive natural numbers

    to avoid exceeding Python's numerical limits.



    Args:

        n: The number of natural numbers to sum.



    Returns:

        The sum of the first n natural numbers.

    """

    if n <= 1:

        return n

    elif n <= 46340:  # Safe limit for Python integer

        return n + recursive_sum_limit_safe(n - 1)

    else:

        return (n * (n + 1)) // 2

