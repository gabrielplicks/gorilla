from typing import List



def skip_every_third(nums: List[int]) -> List[int]:

    """

    Returns a new list with every third element of the original list skipped.



    Args:

        nums: The original list of integers.



    Returns:

        A new list with every third element of the original list skipped.

    """

    result = []



    for i in range(len(nums)):

        if (i + 1) % 3 == 0:

            continue

        result.append(nums[i])



    return result

