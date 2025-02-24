from typing import List



def sum_every_other(nums: List[int]) -> int:

    """Calculates the sum of every other element in a list of numbers.



    Args:

        nums: A list of numbers.



    Raises:

        IndexError: If the list has too few elements (< 2).

    """

    if len(nums) < 2:

        raise IndexError("Too few elements")



    total = 0

    for i in range(0, len(nums), 2):

        total += nums[i]



    return total

