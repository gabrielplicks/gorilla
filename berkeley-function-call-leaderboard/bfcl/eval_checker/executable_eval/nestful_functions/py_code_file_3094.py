from typing import List



def selection_sort_non_local(lst: List[int]) -> List[int]:

    """Sorts a list of integers in ascending order using the selection sort algorithm.

    The function uses non-local variables to keep track of the smallest element and its index.

    Args:

        lst: The list of integers to be sorted.

    """

    for i in range(len(lst)):

        smallest = lst[i]

        smallest_i = i

        for j in range(i + 1, len(lst)):

            if lst[j] < smallest:

                smallest = lst[j]

                smallest_i = j

        lst[smallest_i] = lst[i]

        lst[i] = smallest

    return lst

