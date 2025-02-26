from typing import Dict



def fib_recursive(n: int) -> int:

    """Calculates the nth Fibonacci number using recursion.



    Args:

        n: The position of the Fibonacci number to calculate.



    Returns:

        The nth Fibonacci number.

    """

    if n <= 1:

        return 1

    return fib_recursive(n - 1) + fib_recursive(n - 2)



memo: Dict[int, int] = {}



def fib(n: int) -> int:

    """Calculates the nth Fibonacci number using memoization.



    Args:

        n: The position of the Fibonacci number to calculate.



    Returns:

        The nth Fibonacci number.

    """

    if n in memo:

        return memo[n]

    memo[n] = fib_recursive(n)



    return memo[n]

