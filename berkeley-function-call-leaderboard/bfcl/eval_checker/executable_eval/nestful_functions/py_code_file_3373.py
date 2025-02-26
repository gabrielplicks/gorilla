from typing import List

from collections import Counter



def element_counts(nums: List[int]) -> List[int]:

    """

    Returns a list of the same length as the input list, where each element represents the number of times the corresponding element occurred in the original list.



    Args:

        nums: A list of integers.



    Returns:

        A list of integers.

    """

    counter = Counter(nums)

    return [counter[num] for num in nums]

