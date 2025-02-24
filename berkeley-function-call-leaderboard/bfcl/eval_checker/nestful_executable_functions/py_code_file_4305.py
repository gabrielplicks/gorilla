from typing import List, Dict



def find_sum_indices(nums: List[int], target: int) -> List[int]:

    """Finds the indices of two numbers in an array that add up to a target value.

    Args:

        nums: The array of integers.

        target: The target value.

    """

    lookup: Dict[int, int] = dict()

    for i, num in enumerate(nums):

        lookup[num] = i

    for i, num in enumerate(nums):

        complement = target - num

        if complement in lookup and lookup[complement] != i:

            return [i, lookup[complement]]

    return []

