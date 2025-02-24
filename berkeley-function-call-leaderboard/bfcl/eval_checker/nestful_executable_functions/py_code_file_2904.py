from typing import List



def search_sorted_time_series(array: List[int], timestamp: int) -> int:

    """Searches for a timestamp in a sorted time series array using a binary search algorithm.



    Args:

        array: A sorted time series array.

        timestamp: The timestamp to search for.



    Returns:

        The index of the timestamp in the array or `-1` if the timestamp is not found.

    """

    left = 0

    right = len(array) - 1



    while left <= right:

        mid = (left + right) // 2

        if array[mid] == timestamp:

            while mid > 0 and array[mid-1] == timestamp:

                mid -= 1

            return mid

        elif array[mid] > timestamp:

            right = mid - 1

        else:

            left = mid + 1



    return -1

