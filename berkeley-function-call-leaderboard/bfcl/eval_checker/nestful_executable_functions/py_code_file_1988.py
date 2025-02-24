from typing import Union



def naive_factorial(n: Union[int, None]) -> Union[int, None]:

    """Computes the factorial of a number n using a naive approach.

    Args:

        n: A positive integer.

    """

    if not isinstance(n, int) or n < 0:

        return None  # Invalid input



    result = 1

    for i in range(1, n + 1):

        result *= i

    return result

