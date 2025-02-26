from typing import Union



def factorial_with_int_division(n: Union[int, float]) -> int:

    """Calculates the factorial of a given non-negative integer n using integer division.



    Args:

        n: The non-negative integer for which the factorial is to be calculated.



    Returns:

        The factorial of n.

    """

    if n == 0:

        return 1

    result = 1

    for i in range(1, n + 1):

        result *= i

    return result

