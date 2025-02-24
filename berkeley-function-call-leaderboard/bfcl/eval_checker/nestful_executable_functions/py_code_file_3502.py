from typing import List, Union



def get_largest_element_index(nums: List[Union[int, float]]) -> int:

    """

    Returns the index of the largest element in the list.

    If the list is empty, returns `-1`.

    Raises a `ValueError` if any of the list elements are not numbers.



    Args:

        nums: A list of numbers.

    """

    if not nums:

        return -1



    largest_index = 0

    for i, num in enumerate(nums):

        if not isinstance(num, (int, float)):

            raise ValueError("List elements must be numbers")

        if num > nums[largest_index]:

            largest_index = i



    return largest_index

