from typing import List



def find_max_in_each_subarray(array: List[List[int]]) -> List[int]:

    """Finds the maximum value in each sub-array of a 2D array.



    Args:

        array: A 2D array of integers.



    Returns:

        A new array containing the maximum value from each sub-array.

    """

    if not array:

        return []

    return [max(subarray) for subarray in array]

