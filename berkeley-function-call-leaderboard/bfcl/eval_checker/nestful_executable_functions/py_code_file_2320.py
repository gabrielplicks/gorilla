from typing import List



def find_missing(lower_bound: int, upper_bound: int, l: List[int]) -> List[int]:

    """Finds missing elements in a list of integers within a given range.



    Args:

        lower_bound: The lower bound of the range.

        upper_bound: The upper bound of the range.

        l: The list of integers.

    """

    return [i for i in range(lower_bound, upper_bound + 1) if i not in l]

