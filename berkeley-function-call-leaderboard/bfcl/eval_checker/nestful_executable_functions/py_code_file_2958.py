import random

random.seed(42)
import typing



def random_ints(n: int = 10) -> typing.List[int]:

    """Generates a list of `n` random integers between 0 and 99.



    Args:

        n: The number of random integers to generate. Default is 10.



    Returns:

        A list of `n` random integers between 0 and 99. If `n` is not a

        positive integer between 1 and 100, returns an empty list.

    """

    if not (isinstance(n, int) and 1 <= n <= 100):

        return []



    return [random.randint(0, 99) for _ in range(n)]

