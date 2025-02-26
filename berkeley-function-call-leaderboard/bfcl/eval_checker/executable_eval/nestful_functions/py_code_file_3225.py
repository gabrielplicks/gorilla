import random

random.seed(42)
from typing import List



def generate_unique_random_floats(n: int) -> List[float]:

    """Generates a list of `n` unique random floating point numbers between 0.0 and 1.0.



    The function returns a list of floating point numbers with a precision of 2 decimal places.



    Args:

        n: The number of unique random floating point numbers to generate.

    """

    unique_random_floats = set()

    while len(unique_random_floats) < n:

        unique_random_floats.add(round(random.uniform(0.0, 1.0), 2))

    return list(unique_random_floats)

