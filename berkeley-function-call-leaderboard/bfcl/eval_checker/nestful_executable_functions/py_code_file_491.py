from typing import Union



def even_fibonacci_sum(n: Union[int, float]) -> int:

    """Calculates the sum of even Fibonacci numbers not exceeding a value `n`.



    Args:

        n: The maximum value for the Fibonacci numbers.



    Returns:

        The sum of even Fibonacci numbers not exceeding `n`.

    """

    a, b = 0, 1

    total = 0

    while b <= n:

        if b % 2 == 0:

            total += b

        a, b = b, a + b

    return total

