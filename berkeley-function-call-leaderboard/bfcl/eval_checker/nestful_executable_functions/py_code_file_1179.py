import random

random.seed(42)


def randint(a: int, b: int) -> int:

    """Returns a random integer N such that a <= N <= b.



    Args:

        a: The lower bound of the random integer.

        b: The upper bound of the random integer.

    """

    return random.randint(a, b)

