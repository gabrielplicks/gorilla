from typing import List



def quick_sort(arr: List[int]) -> List[int]:

    """

    Sorts an array using the quick sort algorithm.

    Args:

        arr: The array to be sorted.

    """

    if len(arr) <= 1:  # Base case: the array is already sorted

        return arr



    pivot = arr[0]  # Choose the first element as the pivot

    left = [x for x in arr[1:] if x < pivot]  # Elements less than the pivot

    right = [x for x in arr[1:] if x >= pivot]  # Elements greater than or equal to the pivot



    return quick_sort(left) + [pivot] + quick_sort(right)

