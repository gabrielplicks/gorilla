from typing import List



def insertion_sort_descending(lst: List[int]) -> List[int]:

    """Sorts a list in descending order using the insertion sort algorithm.



    Args:

        lst: The list to be sorted.



    Returns:

        The sorted list.

    """

    for i in range(1, len(lst)):

        key = lst[i]  # Element to be inserted

        j = i - 1

        while j >= 0 and lst[j] < key:

            lst[j + 1] = lst[j]

            j -= 1



        lst[j + 1] = key



    return lst

