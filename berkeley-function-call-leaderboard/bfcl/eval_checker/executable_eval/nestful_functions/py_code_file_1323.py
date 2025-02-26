from typing import List, Tuple



def sort_tuples_by_index(tuples_list: List[Tuple[int, int]], index_to_sort_by: int) -> List[Tuple[int, int]]:

    """Sorts a list of tuples based on the specified index.



    Args:

        tuples_list: The list of tuples to be sorted.

        index_to_sort_by: The index to sort the tuples by.

    """

    return sorted(tuples_list, key=lambda x: x[index_to_sort_by])

