from typing import Dict



fibonacci_cache: Dict[int, int] = {}



def fibonacci(n: int) -> int:

    """Calculates the n-th Fibonacci number using caching to avoid repeated computations.



    Args:

        n: The position of the Fibonacci number to calculate.



    Returns:

        The n-th Fibonacci number.

    """

    if n in fibonacci_cache:

        return fibonacci_cache[n]

    elif n == 0 or n == 1:

        return n

    else:

        fibonacci_cache[n] = fibonacci(n - 1) + fibonacci(n - 2)

        return fibonacci_cache[n]

