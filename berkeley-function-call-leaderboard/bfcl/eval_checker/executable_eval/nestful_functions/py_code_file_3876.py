from typing import List



def sum_less_than_n(n: int, nums: List[int]) -> int:

    """Calculates the sum of all elements in nums that are strictly less than n.

    If no such elements exist, returns -1.

    Args:

        n: The integer to compare against.

        nums: The list of integers to sum.

    """

    sum_less_than_n = 0

    for num in nums:

        if num < n:

            sum_less_than_n += num



    if sum_less_than_n == 0:

        return -1

    else:

        return sum_less_than_n

