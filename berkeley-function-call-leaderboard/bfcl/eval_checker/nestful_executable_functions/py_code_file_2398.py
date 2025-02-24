from typing import List



def disjoint(list1: List[int], list2: List[int]) -> bool:

    """Checks if two lists are disjoint (contain no common elements).



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        True if the lists are disjoint, False otherwise.

    """

    set1 = set(list1)

    set2 = set(list2)

    return len(set1.intersection(set2)) == 0

