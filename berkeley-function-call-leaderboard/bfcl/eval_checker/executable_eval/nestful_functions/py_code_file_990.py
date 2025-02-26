from typing import List



def recursive_binary_search(nums: List[int], target: int) -> int:

    """Recursively performs binary search on a sorted array for a given target.



    Args:

        nums: A sorted list of integers.

        target: The target integer to search for.



    Returns:

        The index of the target in the list if found, otherwise -1.

    """

    if not nums:

        return -1

    mid = len(nums) // 2

    if target == nums[mid]:

        return mid

    elif target < nums[mid]:

        return recursive_binary_search(nums[:mid], target)

    else:

        return recursive_binary_search(nums[mid+1:], target)

