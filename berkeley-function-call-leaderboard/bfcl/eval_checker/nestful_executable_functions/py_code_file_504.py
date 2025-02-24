from typing import List



def sum_of_two(nums: List[int], target: int) -> bool:

    """Checks if there is a pair of numbers in the list that sums to the target number.



    Args:

        nums: A list of numbers.

        target: The target number.



    Returns:

        True if there is a pair of numbers in the list that sums to the target number, and False otherwise.

    """

    for i, num in enumerate(nums):

        for j in range(i + 1, len(nums)):

            if num + nums[j] == target:

                return True

    return False

