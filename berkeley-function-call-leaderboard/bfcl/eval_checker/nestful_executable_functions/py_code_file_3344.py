from typing import Union



def perfect_num(n: Union[int, float]) -> bool:

    """Checks if a positive integer is a perfect number.



    Args:

        n: The positive integer to check.



    Returns:

        True if the number is a perfect number, False otherwise.

    """

    if n <= 0:

        return False

    sum_of_factors = 0

    for i in range(1, n // 2 + 1):

        if n % i == 0:

            sum_of_factors += i

    return sum_of_factors == n

