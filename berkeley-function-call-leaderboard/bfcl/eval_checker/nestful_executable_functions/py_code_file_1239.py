from typing import List



def find_first_match_index(arr1: List[int], arr2: List[int]) -> int:

    """

    Returns the index of the first element that matches in both arrays, or -1 if there is no match.



    Args:

        arr1: The first array.

        arr2: The second array.



    Returns:

        The index of the first matching element, or -1 if there is no match.

    """

    if not arr1 or not arr2:

        return -1



    index = -1

    for i, (x, y) in enumerate(zip(arr1, arr2)):

        if x == y:

            index = i

            break

    return index

