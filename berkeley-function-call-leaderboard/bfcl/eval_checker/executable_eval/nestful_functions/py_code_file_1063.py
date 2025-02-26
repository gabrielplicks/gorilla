from typing import List



def unique_sorted_intersection_list(list1: List[int], list2: List[int]) -> List[int]:

    """

    Find the unique elements in the intersection of two lists.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        A sorted list containing only the unique elements that are present in both lists.

    """

    return sorted({x for x in set(list1) & set(list2)})

