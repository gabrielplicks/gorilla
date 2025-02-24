from typing import Union



def nth_fib_recursion(n: Union[int, float]) -> Union[int, float]:

    """Computes the n-th digit in the Fibonacci sequence.



    Args:

        n: The position of the digit in the Fibonacci sequence.

    """

    if n <= 2:

        return 0 if n == 0 else 1

    return nth_fib_recursion(n - 2) + nth_fib_recursion(n - 1)

