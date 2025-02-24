from typing import List



def largest_three_numbers(nums: List[int]) -> List[int]:

    """

    Returns the largest three numbers in a list of integers.



    Args:

        nums: A list of integers with length 10.



    Raises:

        AssertionError: If the number of elements in the list is less than 3.

    """

    assert len(nums) >= 3, 'The list must have at least 3 elements.'

    return sorted(nums, reverse=True)[:3]

