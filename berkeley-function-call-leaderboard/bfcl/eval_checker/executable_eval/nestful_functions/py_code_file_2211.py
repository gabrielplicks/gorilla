from typing import List



def sum_of_even_ints(nums: List[int] = [0]) -> int:

    """Calculates the sum of even numbers in a list of integers.



    Args:

        nums: A list of integers. Defaults to [0].



    Returns:

        The sum of even numbers in the list.

    """

    return sum([n for n in nums if n % 2 == 0])

