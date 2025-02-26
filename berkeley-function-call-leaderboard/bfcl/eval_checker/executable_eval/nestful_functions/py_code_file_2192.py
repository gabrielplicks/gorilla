from typing import List



def make_lists(numbers: List[int], size: int) -> List[List[int]]:

    """Splits an array of numbers into subarrays of fixed size.



    Args:

        numbers: The array of numbers to split.

        size: The maximum size of each subarray.



    Returns:

        A list of subarrays, each with a maximum size of `size`.

        The last subarray can be shorter than `size`.

        If the original array is empty or `size` is negative,

        the function returns an empty list.

    """

    if not numbers or size < 1:

        return []

    n, r = divmod(len(numbers), size)

    lists = [numbers[i * size:(i + 1) * size] for i in range(n)]

    if r:

        lists.append(numbers[-r:])

    return lists

