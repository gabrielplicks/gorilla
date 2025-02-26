from typing import List



def three_sum_count(nums: List[int], target: int) -> int:

    """

    Counts the number of three distinct elements in a list that can sum to a target.



    Args:

        nums: A list of integers.

        target: The target sum.



    Returns:

        An integer representing the number of three distinct elements that can sum to the target.



    """

    nums.sort()

    count = 0

    for i in range(len(nums) - 2):

        for j in range(i + 1, len(nums) - 1):

            num = target - nums[i] - nums[j]

            if num in nums[j + 1:]:

                count += 1

    return count

