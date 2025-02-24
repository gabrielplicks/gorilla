import random

random.seed(42)
from typing import List



def reservoir_sample(data: List[int], k: int) -> List[int]:

    """

    Returns a random sample of `k` elements from the given list using the reservoir sampling algorithm.

    Args:

        data: The input list.

        k: The number of elements to sample.

    """

    reservoir = []

    for i, elem in enumerate(data):

        if i < k:

            reservoir.append(elem)

        else:

            j = random.randint(0, i)

            if j < k:

                reservoir[j] = elem

    return reservoir

