from typing import List



def indices_of(arr: List[int], target: int) -> List[int]:

    """

    Returns a list of all indices where the target value appears in the array.



    Args:

        arr: A list of integers.

        target: The target value to search for in the array.

    """

    indices = []

    for index, value in enumerate(arr):

        if value == target:

            indices.append(index)

    return indices

