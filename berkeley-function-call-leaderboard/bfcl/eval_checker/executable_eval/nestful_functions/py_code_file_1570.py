from typing import Dict



def fibonacci_dp(n: int) -> int:

    """Calculates the nth number of the Fibonacci sequence using dynamic programming.

    Args:

        n: The position of the number in the Fibonacci sequence.

    """

    memo: Dict[int, int] = {}



    def fib(n: int) -> int:

        """Calculates the nth number of the Fibonacci sequence using dynamic programming.

        Args:

            n: The position of the number in the Fibonacci sequence.

        """

        if n in memo:

            return memo[n]

        if n == 0 or n == 1:

            result = n

        else:

            result = fib(n-1) + fib(n-2)

        memo[n] = result

        return result



    return fib(n)

