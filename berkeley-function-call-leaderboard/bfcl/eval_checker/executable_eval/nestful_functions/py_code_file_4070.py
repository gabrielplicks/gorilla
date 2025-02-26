from typing import Union



def kth_element(arr: Union[list, int], k: int) -> Union[int, None]:

    """

    Returns the kth element of the given list or integer.

    If k is out of bounds, returns the last element.

    Args:

        arr: The list or integer to get the kth element from.

        k: The index of the element to return.

    """

    if k < 0:

        return arr[-1]

    i = 0

    while i < len(arr):

        if i == k:

            return arr[i]

        i += 1

    return arr[-1]

