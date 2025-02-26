from typing import List



def find_prime_composites(n: int) -> List[int]:

    """Finds all prime and composite numbers up to n using bitwise operations.



    Args:

        n: A positive integer.



    Returns:

        A list of all prime and composite numbers up to n, with prime numbers first.

    """

    sieve = [True] * (n + 1)

    for i in range(2, n + 1):

        if sieve[i]:

            for j in range(i * i, n + 1, i):

                sieve[j] = False

    result = [i for i in range(2, n + 1) if sieve[i]] + [i for i in range(2, n + 1) if not sieve[i]]

    return result

