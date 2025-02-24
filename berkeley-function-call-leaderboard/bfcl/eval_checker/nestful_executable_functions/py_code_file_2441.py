from typing import List



def dynamic_sum(nums: List[int]) -> List[int]:

    """Calculates the sum of each integer and the sum of all preceding integers in a list.



    Args:

        nums: A list of integers.



    Returns:

        A list of integers where each element is the sum of the integer itself and the sum of all preceding integers in the original list.

    """

    dp = [0] * len(nums)

    dp[0] = nums[0]

    for i in range(1, len(nums)):

        dp[i] = nums[i] + dp[i - 1]

    return dp

