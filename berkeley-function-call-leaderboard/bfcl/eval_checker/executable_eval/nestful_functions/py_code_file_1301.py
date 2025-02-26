from typing import Union



def fib_iter(n: Union[int, float]) -> Union[int, float]:

    """Calculates the n-th Fibonacci number using the iterative approach.



    Args:

        n: The position of the Fibonacci number to calculate.



    Returns:

        The n-th Fibonacci number.

    """

    a, b = 0, 1

    if n <= 0:

        return 0

    for _ in range(n-1):

        a, b = b, a+b

    return b

