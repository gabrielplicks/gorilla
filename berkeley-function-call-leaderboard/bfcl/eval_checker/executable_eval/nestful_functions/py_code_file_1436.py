from typing import List, Union



def linear_search(arr: List[Union[int, float]], target: Union[int, float]) -> int:

    """

    Performs a linear search to find the index of a target value in an unsorted array.

    If the value is not found in the array, returns -1.



    Args:

        arr: The array to search.

        target: The target value to find.

    """

    for i, val in enumerate(arr):

        if val == target:

            return i

    return -1

