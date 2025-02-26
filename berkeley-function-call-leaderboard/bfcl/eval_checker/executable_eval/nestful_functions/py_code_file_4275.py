from typing import List



def maxSubarraySum(nums: List[int]) -> int:

    """Calculates the maximum sum of all non-empty sublists in a list of integers.



    Args:

        nums: A list of integers.



    Returns:

        The maximum sum of all non-empty sublists.

    """

    max_sum = nums[0]

    curr_max = nums[0]

    for num in nums[1:]:

        curr_max = max(num, curr_max + num)

        max_sum = max(max_sum, curr_max)

    return max_sum

