from typing import List



def count_non_zeros(nums: List[int]) -> int:

    """Counts the number of non-zero integers in a list.



    Args:

        nums: A list of integers.



    Returns:

        The number of non-zero integers in the list.

    """

    count = 0

    for num in nums:

        if num != 0:

            count += 1

    return count

