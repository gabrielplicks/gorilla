from typing import List



def find_first_of(arr: List[int], target: int) -> int:

    """Finds the index of the first occurrence of the target value in the array.



    Args:

        arr: The array of integers.

        target: The target value to find.



    Returns:

        The index of the first occurrence of the target value in the array, or -1 if the target is not present.

    """

    for i, e in enumerate(arr):

        if e == target:

            return i

    return -1

