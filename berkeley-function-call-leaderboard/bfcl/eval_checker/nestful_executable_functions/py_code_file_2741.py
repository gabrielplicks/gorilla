from typing import List



def find_pivot(sorted_array: List[int]) -> int:

    """Finds the pivot index in a rotated sorted array.



    Args:

        sorted_array: A sorted array that may be rotated.



    Returns:

        The index of the pivot if the array is rotated, or -1 if the array is not rotated.

    """

    left, right = 0, len(sorted_array) - 1



    while left <= right:

        mid = (left + right) // 2



        if mid > 0 and sorted_array[mid] < sorted_array[mid - 1]:

            return mid



        if sorted_array[mid] >= sorted_array[left]:

            left = mid + 1

        else:

            right = mid - 1



    return -1

