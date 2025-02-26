import random

random.seed(42)
import typing



def generate_random_sum(n: int) -> typing.Tuple[int, int]:

    """Generates two random numbers `x` and `y` such that `x + y` is equal to a given number `n`.



    Args:

        n: The given number.



    Returns:

        A tuple of two integers `x` and `y`.

    """

    x = random.randrange(0, n + 1)

    y = n - x

    return x, y

