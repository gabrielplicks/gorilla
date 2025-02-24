from typing import List



def find_first_duplicate_element(arr: List[int]) -> int:

    """Finds and returns the index of the first duplicate element in a given array of integers.

    Returns -1 if there are no duplicate elements.

    Args:

        arr: A list of integers.

    """

    for i, num in enumerate(arr):

        for j in range(i + 1, len(arr)):

            if num == arr[j]:

                return i

    return -1

