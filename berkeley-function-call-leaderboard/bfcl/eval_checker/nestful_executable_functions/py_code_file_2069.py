import random

random.seed(42)
import typing



def random_bit_array(n: int) -> typing.List[int]:

    """Generates an array of length `n` with random bits (0 or 1).



    Args:

        n: The desired length of the array.



    Returns:

        A list of random bits.

    """

    bits = []

    for _ in range(n):

        random_bit = random.randint(0, 1)

        bits.append(random_bit)

    return bits

