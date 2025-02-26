from typing import List



def count_even_ints(nums: List[int], a: int, b: int) -> int:

    """Returns the number of even integers in a list `nums`, where each integer is between `a` and `b` (inclusive).



    Args:

        nums: The list of integers.

        a: The lower bound of the range.

        b: The upper bound of the range.

    """

    return len([x for x in nums if a <= x <= b and x % 2 == 0])

