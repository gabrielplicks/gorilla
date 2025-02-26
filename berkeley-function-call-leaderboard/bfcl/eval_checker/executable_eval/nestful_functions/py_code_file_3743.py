from typing import List



def brute_force_max_index(arr: List[int]) -> int:

    """Finds the index of the maximum element in an array using the brute force algorithm.



    Args:

        arr: The input array.



    Returns:

        The index of the maximum element in the array.

    """

    max_idx = 0

    for i in range(len(arr)):

        if arr[i] > arr[max_idx]:

            max_idx = i

    return max_idx

