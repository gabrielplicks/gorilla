from typing import List



def first_few_primes(n: int) -> List[int]:

    """Returns a list of the first few prime numbers.



    Args:

        n: The number of primes to return.



    Returns:

        A list of the first few prime numbers.

    """

    if not isinstance(n, int) or n < 0:

        return []



    primes = []

    i = 2

    while len(primes) < n:

        is_prime = True

        for j in range(2, i):

            if i % j == 0:

                is_prime = False

                break

        if is_prime:

            primes.append(i)

        i += 1

    return primes

