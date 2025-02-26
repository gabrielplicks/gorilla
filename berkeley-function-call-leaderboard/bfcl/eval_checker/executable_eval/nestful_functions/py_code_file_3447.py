import random

random.seed(42)
from typing import List



def shuffle_sort(lst: List[float], p: float) -> List[float]:

    """Shuffles and sorts a list based on a given probability.



    Args:

        lst: The list to be shuffled and sorted.

        p: The probability of shuffling the list.



    Returns:

        The sorted list.

    """

    if p == 0:

        return sorted(lst)

    elif p == 1:

        random.shuffle(lst)

        return sorted(lst)

    else:

        r = random.random()

        if r <= p:

            random.shuffle(lst)

            return sorted(lst)

        else:

            return sorted(lst)

