from typing import Union



def factorial_trailing_zeros(n: Union[int, float]) -> int:

    """Calculates the number of trailing zeros in the factorial of n.



    Args:

        n: The number to calculate the factorial of.



    Raises:

        TypeError: If n is not a number.

    """

    if not isinstance(n, int):

        raise TypeError('n must be an integer')



    factorial = 1

    for i in range(1, n + 1):

        factorial *= i



    trailing_zeros = 0

    while factorial % 10 == 0:

        trailing_zeros += 1

        factorial //= 10



    return trailing_zeros

