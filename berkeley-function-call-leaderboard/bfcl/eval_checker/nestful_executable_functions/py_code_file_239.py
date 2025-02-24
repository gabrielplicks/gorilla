from typing import List



def is_consecutive_ints(nums: List[int]) -> bool:

    """Checks if a given list of integers is composed of consecutive integers.



    Args:

        nums: A list of integers.



    Returns:

        True if the list is consecutive, False otherwise.

    """

    nums.sort()

    for i in range(len(nums) - 1):

        if nums[i + 1] - nums[i] != 1:

            return False

    return True

