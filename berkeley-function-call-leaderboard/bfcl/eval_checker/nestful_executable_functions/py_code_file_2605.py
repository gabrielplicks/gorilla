from typing import List, Tuple



def find_min_and_index(arr: List[int]) -> Tuple[int, int]:

    """Finds the minimum value and its index in an array.



    Args:

        arr: A list of integers.



    Returns:

        A tuple containing the minimum value and its index.

    """

    if not arr:

        return ()

    min_val = arr[0]

    min_index = 0

    for i in range(1, len(arr)):

        if arr[i] < min_val:

            min_val = arr[i]

            min_index = i

    return (min_val, min_index)

