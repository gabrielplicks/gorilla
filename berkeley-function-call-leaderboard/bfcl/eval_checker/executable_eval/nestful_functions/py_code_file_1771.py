from typing import List



def move_first_n_to_end(nums: List[int], n: int) -> List[int]:

    """Moves the first `n` elements of a list to the end.



    Args:

        nums: The input list of numbers.

        n: The number of elements to move to the end.



    Returns:

        The modified list with the first `n` elements moved to the end.

    """

    result = []

    for i in range(n, len(nums)):

        result.append(nums[i])

    for i in range(0, n):

        result.append(nums[i])

    return result

