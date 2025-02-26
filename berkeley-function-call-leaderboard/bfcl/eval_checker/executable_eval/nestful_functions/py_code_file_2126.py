from functools import reduce

from typing import List



def sum_or_list(nums: List[int]) -> str:

    """

    Calculates the sum of a list of integers and returns it as a string.

    If the sum is greater than 2**31 - 1, it converts the sum to a list of integers and returns the string representation of this list.



    Args:

        nums: A list of integers.



    Returns:

        A string representation of the sum or the list of integers.

    """

    total = reduce(lambda x, y: x + y, nums)



    if total > 2**31 - 1:

        total = [int(c) for c in str(total)]

    return str(total)

