from typing import Dict



fib_cache: Dict[int, int] = {0: 0, 1: 1, 2: 1}



def find_nth_fibonacci_memoized(n: int) -> int:

    """Returns the nth Fibonacci number using memoization.



    Args:

        n: The position of the Fibonacci number to find.



    Returns:

        The nth Fibonacci number.

    """

    if n < 0:

        n = len(fib_cache) + n



    if n in fib_cache:

        return fib_cache[n]



    if n not in fib_cache:

        fib_cache[n] = find_nth_fibonacci_memoized(n-1) + find_nth_fibonacci_memoized(n-2)



    return fib_cache[n]

