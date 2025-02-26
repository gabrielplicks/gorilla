from typing import List



def has_pair_with_negative_product(nums: List[int]) -> bool:

    """Checks if there are two integers in the list whose product is negative.

    Args:

        nums: A list of unique integers.

    """

    for i, x in enumerate(nums):

        for y in nums[i + 1:]:

            if x * y < 0:

                return True

    return False

