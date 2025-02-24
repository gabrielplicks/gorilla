from typing import Union



def int_factorial(n: Union[int, float]) -> int:

    """Calculates the factorial of an integer `n` ≥ 0.



    Args:

        n: The integer to calculate the factorial of.



    Raises:

        TypeError: If `n` is not an integer.

    """

    if not isinstance(n, int):

        raise TypeError("Argument must be an integer")



    if n < 0:

        return 0

    elif n == 0:

        return 1

    else:

        factorial = 1

        for i in range(1, n + 1):

            factorial *= i

        return factorial

