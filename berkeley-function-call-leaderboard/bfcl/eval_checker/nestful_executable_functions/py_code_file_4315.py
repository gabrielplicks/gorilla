from typing import List



def has_intersection(list1: List[int], list2: List[int]) -> bool:

    """Checks if two lists have any common elements using for loops and conditional evaluation.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        True if the lists have any common elements, otherwise False.

    """

    for value in list1:

        if value in list2:

            return True



    return False

