from typing import List



def get_max_index(nums: List[int]) -> int:

    """Finds the index of the maximum value in a list of numbers.

    Args:

        nums: A list of numbers.

    Returns:

        The index of the maximum value in the list, or -1 if the list is empty or contains only one element.

    """

    if len(nums) < 2:

        return -1

    return nums.index(max(nums))

