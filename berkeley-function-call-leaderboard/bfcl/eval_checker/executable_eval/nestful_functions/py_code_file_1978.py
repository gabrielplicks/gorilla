from typing import Union



def calculate_factorial(n: Union[int, float]) -> int:

    """Calculates the factorial of a number using a for loop.



    Args:

        n: The number to calculate the factorial of.

    """

    factorial = 1

    for i in range(1, int(n) + 1):

        factorial *= i

    return factorial

