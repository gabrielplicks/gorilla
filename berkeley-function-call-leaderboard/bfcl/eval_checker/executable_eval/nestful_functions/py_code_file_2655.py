from typing import Union



def factorial_recursively(n: Union[int, float]) -> Union[int, float]:

    """Calculates the factorial of a positive integer `n` using recursion.



    Args:

        n: A positive integer.



    Returns:

        The factorial of `n`.

    """

    if n <= 1:

        return 1

    return n * factorial_recursively(n - 1)

