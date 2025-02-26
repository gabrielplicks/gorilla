from typing import List



def find_kth_largest(nums: List[int], k: int) -> int:

    """Finds the kth largest element in a list of integers.



    Args:

        nums: A list of integers.

        k: A positive integer.



    Returns:

        The kth largest element in the list. If k is not valid, returns None.

    """

    if k <= 0 or k > len(nums):

        return None

    nums.sort(reverse=True)

    return nums[k - 1]

