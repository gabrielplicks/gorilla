import random

random.seed(42)


def generate_pair_of_numbers(random_seed: int = None) -> tuple:

    """

    Function to generate a pair of numbers x and y that are uniformly random and are in the range of [0, 1000000000].

    Args:

        random_seed: (optional) a seed for the random number generator. If not provided, a random seed will be generated.

    """

    if random_seed is None:

        random_seed = random.randint(0, 1000000000)

    random.seed(random_seed)

    x = random.randrange(0, 1000000000)

    y = random.randrange(0, 1000000000)

    return x, y

