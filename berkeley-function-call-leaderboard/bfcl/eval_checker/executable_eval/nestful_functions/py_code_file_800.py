from typing import List, Set



def find_smallest_positive_missing(nums: List[int]) -> int:

    """Finds the smallest positive integer that is not present in the given list.



    Args:

        nums: A list of integers.



    Returns:

        The smallest positive integer that is not present in the list.

    """

    s = set(nums)

    i = 1

    while i in s:

        i += 1

    return i

