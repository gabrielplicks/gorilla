from typing import List, Union



def get_min_and_max(nums: List[Union[int, float]]) -> Union[None, tuple]:

    """

    Returns the smallest and largest numbers in a list of numbers.

    If the list contains a non-numeric value, returns None.

    If the list is empty, returns (None, None).



    Args:

        nums: A list of numbers (integers or floats).

    """

    if not all(isinstance(n, (int, float)) for n in nums):

        return None

    if not nums:

        return None, None

    return min(nums), max(nums)

