from typing import List



def longest_consecutive_sequence(nums: List[int]) -> int:

    """Identifies the longest consecutive sequence of integers (at least 3) in a given list of integers, and returns its length.



    Args:

        nums: The input list of integers.



    Returns:

        The length of the longest consecutive sequence found.

    """

    if len(nums) < 3:

        return 0



    current_length = 1

    max_length = 1



    for i in range(1, len(nums)):

        if nums[i] == nums[i - 1] + 1:

            current_length += 1

        else:

            max_length = max(current_length, max_length)

            current_length = 1



    return max(current_length, max_length)

