import random

random.seed(42)
from typing import List



def create_random_integers(low: int, high: int, size: int) -> List[int]:

    """Creates a list of random integers of given range and size.



    Args:

        low: The lower limit of the range.

        high: The upper limit of the range.

        size: The number of integers to generate.



    Returns:

        A list of randomly generated integers of specified size, where each integer is within the specified range.

    """

    random_ints = []

    for _ in range(size):

        random_ints.append(random.randint(low, high))

    return random_ints

