import sys



def factorial_bitwise(n: int) -> int:

    """Calculates the factorial of a given number using bitwise operators and recursion.



    Args:

        n: The number to calculate the factorial of.



    Returns:

        The factorial of the given number.

    """

    assert isinstance(n, int) and n >= 0, "Input must be a non-negative integer"

    assert n <= sys.maxsize, "Input is too large"



    if n == 0:

        return 1

    if n == 1:

        return 1



    return n * factorial_bitwise(n - 1)

