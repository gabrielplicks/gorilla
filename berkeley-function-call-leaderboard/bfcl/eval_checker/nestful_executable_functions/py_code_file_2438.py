from typing import List, Union



def max_of_list(nums: List[Union[int, float]]) -> Union[int, float, None]:

    """Returns the maximum element in a list of numbers. If the list is empty, returns None.



    Args:

        nums: A list of numbers.

    """

    return max(nums, default=None)

