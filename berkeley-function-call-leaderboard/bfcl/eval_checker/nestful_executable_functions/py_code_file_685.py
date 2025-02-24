from typing import List



def diff_adjacent(nums: List[int]) -> List[int]:

    """

    Returns a new list containing the difference between each element and its adjacent element.

    If an element has no adjacent element, its corresponding new element should be 0.



    Args:

        nums: A list of integers.

    """

    return [abs(nums[i] - nums[i+1]) if i < len(nums) - 1 else 0 for i in range(len(nums))]

