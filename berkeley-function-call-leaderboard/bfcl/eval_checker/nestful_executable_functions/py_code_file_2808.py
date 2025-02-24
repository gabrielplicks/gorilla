from typing import List



def sum_even_fibonacci_numbers(n: int) -> int:

    """Calculates the sum of consecutive even Fibonacci numbers until a given number is reached.



    Args:

        n: The upper bound for the Fibonacci numbers.



    Returns:

        The sum of the even Fibonacci numbers.

    """

    a, b = 0, 1

    fibonacci_numbers: List[int] = []



    while a <= n:

        if a % 2 == 0:

            fibonacci_numbers.append(a)

        a, b = b, a + b



    return sum(fibonacci_numbers)

