from typing import Union



def sum_of_squares_of_odd_numbers(n: Union[int, float]) -> int:

    """Calculates the sum of the squares of naturally ordered odd numbers from 1 to n (inclusive).



    Args:

        n: A positive integer.



    Returns:

        The sum of the squares of the naturally ordered odd numbers from 1 to n (inclusive).

    """

    sum_of_squares = 0

    for i in range(1, int(n) + 1):

        if i % 2 == 1:  # Check if i is odd

            sum_of_squares += i ** 2

    return sum_of_squares

