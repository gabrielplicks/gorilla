from typing import List



def sort_and_exclude_first_and_last(nums: List[int]) -> List[int]:

    """Sorts an integer array in ascending order and returns the array without the first and last elements.

    If there are fewer than 2 elements, returns an empty list.



    Args:

        nums: The integer array to be sorted.



    Returns:

        The sorted array without the first and last elements.

    """

    if len(nums) < 2:

        return []



    nums.sort()

    return nums[1:-1]

