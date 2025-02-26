import math



def generate_random_number_from_seed(timestamp: float, seed: int) -> int:

    """Generates a random number from a given seed.

    Args:

        timestamp: The timestamp value.

        seed: The seed value.

    """

    digits = str(timestamp).split('.')[-1]

    quotient = math.floor(int(digits) / seed)

    return int(digits) % seed

