from typing import List



def find_prime_less_than_100(numbers: List[int]) -> List[int]:

    """

    Returns a list of all the prime numbers that are in the input list and are also less than 100.



    Args:

        numbers: A list of integers.



    Returns:

        A list of prime numbers that are in the input list and are also less than 100.

    """

    return [n for n in numbers if is_prime(n) and n < 100]



def is_prime(n: int) -> bool:

    """

    Checks if a number is prime.



    Args:

        n: The number to check.



    Returns:

        True if the number is prime, False otherwise.

    """

    for i in range(2, int(n**0.5)+1):

        if n % i == 0:

            return False

    return True

