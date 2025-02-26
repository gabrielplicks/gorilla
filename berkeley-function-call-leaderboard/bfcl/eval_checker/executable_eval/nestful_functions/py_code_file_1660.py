from typing import List, Optional



def index_of_min(nums: List[int]) -> Optional[int]:

    """

    Returns the index of the smallest element in a list of integers.

    If the list is empty or contains only 1 element, returns None.



    Args:

        nums: A list of integers.



    Returns:

        The index of the smallest element in the list, or None if the list is empty or contains only 1 element.

    """

    if len(nums) <= 1:

        return None

    min_index = 0

    for i, num in enumerate(nums):

        if num < nums[min_index]:

            min_index = i

    return min_index

