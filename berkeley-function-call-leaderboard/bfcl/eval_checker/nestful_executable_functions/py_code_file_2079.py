from typing import List



def in_place_sort(lst: List[int]) -> List[int]:

    """Sorts a list of integers in ascending order using an in-place sorting algorithm.



    The algorithm iterates through the list elements from the beginning to the end.

    For each element, it finds the smallest element among all the elements to its right.

    It then swaps the element with the smallest element found in the previous step.



    Args:

        lst: The list of integers to be sorted.



    Returns:

        The sorted list of integers.

    """

    for i in range(len(lst)):

        min_idx = i

        for j in range(i+1, len(lst)):

            if lst[j] < lst[min_idx]:

                min_idx = j

        lst[i], lst[min_idx] = lst[min_idx], lst[i]

    return lst

