import random

random.seed(42)


def random_between(min_val: float = 0, max_val: float = 1) -> float:

    """

    Generate a random number between the given range of min_val and max_val.



    Usage:

    random_between() # Generates a random number between 0 and 1

    random_between(5, 10) # Generates a random number between 5 and 10



    Args:

        min_val: The minimum value of the range. Defaults to 0.

        max_val: The maximum value of the range. Defaults to 1.

    """

    return min_val + (max_val - min_val) * random.random()

