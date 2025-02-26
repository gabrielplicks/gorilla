from typing import List, Tuple



def next_greatest(n: int, nums: List[int]) -> List[Tuple[int, int]]:

    """

    Returns a list of tuples with each tuple containing the element at index `i` and the next greatest element in `nums`.

    If no such next greatest element exists, the second element should be -1.



    Args:

        n: The length of the input list.

        nums: The input list of integers.



    Returns:

        A list of tuples.

    """

    result = []

    for i, num in enumerate(nums):

        next_greater = -1

        for j in range(i + 1, len(nums)):

            if nums[j] > num:

                next_greater = nums[j]

                break

        result.append((num, next_greater))

    return result

