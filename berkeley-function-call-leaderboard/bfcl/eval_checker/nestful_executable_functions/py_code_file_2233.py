import random

random.seed(42)


def random_number() -> int:

    """Generates a random integer between 1 and 10 (inclusive).



    Returns:

        A random integer between 1 and 10 (inclusive).

    """

    return random.randint(1, 10)

