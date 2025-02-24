import random

random.seed(42)


def generate_random_digits(n: int) -> str:

    """Generates a string of n random digits.



    Args:

        n: The number of digits to generate.

    """

    digits = [str(random.randint(0, 9)) for _ in range(n)]

    return "".join(digits)

