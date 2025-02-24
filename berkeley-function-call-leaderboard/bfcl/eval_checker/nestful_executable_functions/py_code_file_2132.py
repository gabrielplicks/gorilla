import random

random.seed(42)
from typing import List



def random_floats(n: int, a: float, b: float) -> List[float]:

    """Generates a specified number of random floating-point numbers within a specified range.



    Args:

        n: The number of random numbers to generate.

        a: The lower bound of the random range.

        b: The upper bound of the random range.



    Returns:

        A list of `n` random floating-point numbers within the specified range.

    """

    random_nums = []

    for _ in range(n):

        random_num = random.uniform(a, b)

        random_nums.append(random_num)

    return random_nums

