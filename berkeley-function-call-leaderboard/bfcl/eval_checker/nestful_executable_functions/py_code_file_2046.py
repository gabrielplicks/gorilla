from typing import List



def is_all_pos_neg(nums: List[float]) -> bool:

    """Checks if all numbers in the list are either all positive or all negative.



    Args:

        nums: A list of numbers.



    Returns:

        True if all numbers in the list are either all positive or all negative, False otherwise.

    """

    return all(num > 0 for num in nums) or all(num < 0 for num in nums)

