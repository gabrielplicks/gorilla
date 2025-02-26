from typing import List



def list_operations(list1: List[int], list2: List[int]) -> List[List[int]]:

    """Calculates the intersection, union, and difference between two lists.

    Args:

        list1: The first list.

        list2: The second list.

    """

    set1 = set(list1)

    set2 = set(list2)

    intersection = set1.intersection(set2)

    union = set1.union(set2)

    difference = set1.difference(set2)



    return [list(intersection), list(union), list(difference)]

