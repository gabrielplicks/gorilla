from typing import List



def both_positive_and_negative(nums: List[int]) -> bool:

    """Checks if a list of integers contains both positive and negative integers.



    Args:

        nums: A list of integers.



    Returns:

        True if the list contains both positive and negative integers, False otherwise.

    """

    if len(nums) == 0:

        return False



    positive_exists = False

    negative_exists = False



    for num in nums:

        if num > 0:

            positive_exists = True

        elif num < 0:

            negative_exists = True



        if positive_exists and negative_exists:

            return True



    return False

