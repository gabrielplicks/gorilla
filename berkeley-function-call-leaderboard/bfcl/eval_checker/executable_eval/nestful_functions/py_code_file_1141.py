from typing import List



def shift_list(lst: List[int], n: int) -> List[int]:

    """Shifts the elements of a list by `n` indexes.

    Args:

        lst: The input list of numbers.

        n: The number of indexes to shift the elements.

    """

    return lst[n:] + lst[:n]

