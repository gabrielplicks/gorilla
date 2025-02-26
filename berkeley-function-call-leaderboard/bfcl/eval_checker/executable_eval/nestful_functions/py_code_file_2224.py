from typing import List



def largest_contiguous_subarray_sum(nums: List[int]) -> int:

    """Finds the contiguous subarray with the largest sum in a given array of integers.



    Args:

        nums: A list of integers.



    Returns:

        The maximum sum of a contiguous subarray in the given array.

    """

    current_sum = 0

    max_sum = 0

    for num in nums:

        current_sum = max(0, current_sum + num)

        max_sum = max(max_sum, current_sum)

    return max_sum

