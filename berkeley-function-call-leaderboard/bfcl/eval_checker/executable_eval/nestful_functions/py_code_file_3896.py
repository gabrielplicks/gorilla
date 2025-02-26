from typing import Union



def is_prime_naive(n: Union[int, float]) -> bool:

    """Checks if a given number is a prime number.



    Args:

        n: The number to check.

    """

    for i in range(2, n):

        if n % i == 0:

            return False

    return True

