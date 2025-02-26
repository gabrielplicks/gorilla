from typing import List



def merge_recursive(sorted_list_1: List[int], sorted_list_2: List[int]) -> List[int]:

    """Merges two sorted lists into a single sorted list using recursion.

    Args:

        sorted_list_1: The first sorted list.

        sorted_list_2: The second sorted list.

    """

    if not sorted_list_1:

        return sorted_list_2

    elif not sorted_list_2:

        return sorted_list_1

    if sorted_list_1[0] < sorted_list_2[0]:

        return [sorted_list_1[0]] + merge_recursive(sorted_list_1[1:], sorted_list_2)

    else:

        return [sorted_list_2[0]] + merge_recursive(sorted_list_1, sorted_list_2[1:])

