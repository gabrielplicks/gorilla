from typing import List



def get_next_permutation(nums: List[int]) -> List[int]:

    """

    Returns the next lexicographically greater permutation of the input list.

    If the input list is already the last permutation, returns the list in ascending order.



    Args:

        nums: A list of distinct integers.

    """

    i = len(nums) - 2

    while i >= 0 and nums[i] >= nums[i + 1]:

        i -= 1

    if i == -1:

        return nums[::-1]

    j = len(nums) - 1

    while nums[j] <= nums[i]:

        j -= 1

    nums[i], nums[j] = nums[j], nums[i]

    nums[i + 1:] = nums[i + 1:][::-1]



    return nums

