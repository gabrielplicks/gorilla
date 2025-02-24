from typing import List



def add_list_elements(nums: List[int]) -> int:

    """Calculates the sum of the elements in a list of numbers.

    Args:

        nums: A list of numbers.

    """

    if len(nums) == 0:

        return 0

    return sum(nums)

