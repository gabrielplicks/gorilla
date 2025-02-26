from typing import Union



def difference_sum(n: Union[int, float]) -> int:

    """Calculates the difference between the sum of the squares of the first n natural numbers and the square of the sum.



    Args:

        n: The number of natural numbers to consider.



    Returns:

        The difference between the sum of the squares and the square of the sum.

    """

    sum_of_squares = 0

    square_of_sum = 0

    for i in range(1, n + 1):

        sum_of_squares += i ** 2

        square_of_sum += i

    return (square_of_sum ** 2) - sum_of_squares

