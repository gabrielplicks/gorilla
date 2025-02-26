from typing import List



def divide_without_remainder(n: int) -> List[int]:

    """Finds all the numbers that divide a given number without a remainder.



    Args:

        n: The number to find divisors for.



    Returns:

        A list of all the numbers that divide the given number without a remainder.

    """

    result = []

    for i in range(1, n + 1):

        if n % i == 0:

            result.append(i)

    return result

