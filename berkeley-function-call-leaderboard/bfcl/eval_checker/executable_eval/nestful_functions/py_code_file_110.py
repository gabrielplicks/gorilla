import random

random.seed(42)


def gen_random_hex(n: int) -> str:

    """Generates a random hexadecimal string of length n.



    Args:

        n: The desired length of the hexadecimal string.



    Returns:

        A random hexadecimal string of length n.

    """

    random_hex_digits = [hex(random.randint(0, 15))[2:] for _ in range(n)]

    return ''.join(random_hex_digits)

