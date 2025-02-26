from typing import List



def find_first_geq(nums: List[int]) -> int:

    """Finds the index of the first element that is greater or equal to the maximum value in a list of integers.



    Args:

        nums: A list of integers.



    Returns:

        The index of the first element that is greater or equal to the maximum value in the list. If there is no such element, returns -1.

    """

    max_val = max(nums)

    for i, num in enumerate(nums):

        if num >= max_val:

            return i

    return -1

