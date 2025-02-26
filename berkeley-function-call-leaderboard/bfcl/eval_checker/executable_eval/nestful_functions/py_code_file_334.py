from typing import List



def group_by_parity(nums: List[int]) -> tuple:

    """Groups a list of integers by parity.



    Args:

        nums: A list of integers.



    Returns:

        A tuple containing two lists: one with all the even integers, the other with all the odd integers.

        The elements are sorted within their respective lists.

    """

    even, odd = [], []

    for num in nums:

        if num % 2 == 0:

            even.append(num)

        else:

            odd.append(num)

    return sorted(even), sorted(odd)

