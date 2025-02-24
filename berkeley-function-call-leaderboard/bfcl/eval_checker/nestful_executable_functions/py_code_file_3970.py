from typing import List



def sort_sequence(nums: List[int]) -> List[int]:

    """Sorts a sequence of integers based on the following condition.

    If a number is odd, it should be sorted in ascending order,

    and if it is even, it should be sorted in descending order.



    Args:

        nums: The sequence of integers to be sorted.



    Returns:

        The sorted sequence of integers.

    """

    odd_nums = []

    even_nums = []



    for num in nums:

        if num % 2 == 1:

            odd_nums.append(num)

        else:

            even_nums.append(num)



    odd_nums.sort()

    even_nums.sort(reverse=True)



    return odd_nums + even_nums

