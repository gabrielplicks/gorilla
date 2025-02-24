from typing import List



def is_ascending(nums: List[int]) -> bool:

    """Determines whether the given list of numbers `nums` is in ascending order.



    Args:

        nums: The list of numbers to check.



    Returns:

        `True` if the list is in ascending order, and `False` otherwise.

    """

    if nums is None:

        return False

    if len(nums) <= 1:

        return True

    for i in range(1, len(nums)):

        if nums[i] < nums[i - 1]:

            return False

    return True

