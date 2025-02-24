from typing import List



def binary_search_leftmost(lst: List[int], value: int) -> int:

    """Finds the first (leftmost) index of a value in a sorted list of integers.



    Args:

        lst: A sorted list of integers.

        value: The value to search for.



    Returns:

        The first (leftmost) index of the value in the list, or -1 if the value is not found.

    """

    left = 0

    right = len(lst) - 1



    while left <= right:

        mid = (left + right) // 2

        if lst[mid] == value:

            if mid == 0 or lst[mid-1] < value:

                return mid

            else:

                right = mid - 1

        elif lst[mid] < value:

            left = mid + 1

        else:

            right = mid - 1

    return -1

