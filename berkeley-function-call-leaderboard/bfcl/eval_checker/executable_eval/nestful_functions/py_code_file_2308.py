from random import randint

import random
random.seed(42)
from typing import List, Tuple



def partition_list(lst: List, k: int = None) -> Tuple[List, List]:

    """Partitions a list into two parts.



    The first part has a length of `k`, where `k` is a random integer between 0 and `len(lst) - 1`, inclusive.

    The second part contains the remaining elements of the list. If the function is not provided a value for `k`,

    then `k` is randomly chosen.



    Args:

        lst: The list to be partitioned.

        k: The index to partition the list. If not provided, a random index is chosen.



    Returns:

        A tuple containing the two resulting lists.

    """

    if k is None:

        k = randint(0, len(lst) - 1)

    first_part = lst[:k]

    second_part = lst[k:]



    return first_part, second_part

