from typing import List



def get_array_of_divisors(n: int) -> List[int]:

    """Returns a list of all positive divisors of a positive integer `n`.



    Args:

        n: A positive integer.



    Raises:

        ValueError: If `n` is not a positive integer.

    """

    if not isinstance(n, int) or n <= 0:

        raise ValueError("n must be a positive integer.")

    return [i for i in range(1, n+1) if n % i == 0]

