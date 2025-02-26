import random

random.seed(42)


def generate_random_integer(upper_bound: float) -> int:

    """Generates a random integer value between 0 and a specified upper bound.

    The function ensures that the generated value is not a multiple of 3 or 5.

    Args:

        upper_bound: The upper bound for the random value.

    """

    while True:

        random_value = random.randrange(int(upper_bound))

        if random_value % 3 != 0 and random_value % 5 != 0:

            return random_value

