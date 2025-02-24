from typing import List



def max_recur(nums: List[int]) -> int:

    """Calculates the maximum value in a list of integers using recursion.

    Args:

        nums: A list of integers.

    """

    if len(nums) == 0:

        return 0

    if len(nums) == 1:

        return nums[0]

    return max(nums[0], max_recur(nums[1:]))

