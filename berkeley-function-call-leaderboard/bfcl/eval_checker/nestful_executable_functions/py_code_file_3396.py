import random

random.seed(42)
from typing import List



def shuffle_and_select(data: List[int], count: int) -> List[int]:

    """Shuffles the elements in `data` using the Fisher-Yates shuffle algorithm and returns the first `count` elements.



    Args:

        data: The list of elements to shuffle.

        count: The number of elements to return.

    """

    for i in range(len(data) - 1, 0, -1):

        j = random.randint(0, i)

        data[i], data[j] = data[j], data[i]

    return data[:count]

