from typing import List



def next_positive(arr: List[int]) -> int:

    """

    Returns the first positive integer that is not in the list.

    If all integers are positive, returns the next positive integer.

    If there are no positive integers in the list, returns 1.



    Args:

        arr: A non-empty list of integers.



    Raises:

        ValueError: If the input is not a non-empty list of integers.

    """

    if not isinstance(arr, list) or not all(isinstance(i, int) for i in arr) or not arr:

        raise ValueError("Input must be a non-empty list of integers.")



    arr.sort()

    for i, num in enumerate(arr):

        if num <= 0:

            return i + 1

    return arr[-1] + 1 if arr else 1

