from typing import Tuple



def fibonacci_sum_divisible_by_2(max_value: int) -> int:

    """Calculates the sum of all Fibonacci numbers that are divisible by 2 and less than max_value.



    Args:

        max_value: The maximum value for the Fibonacci sequence.



    Returns:

        The sum of all Fibonacci numbers that are divisible by 2 and less than max_value.

    """

    a, b = 0, 1

    total = 0



    while a < max_value:

        if a % 2 == 0:

            total += a

        a, b = b, a + b



    return total

