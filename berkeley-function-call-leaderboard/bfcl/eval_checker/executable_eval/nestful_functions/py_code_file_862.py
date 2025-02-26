from typing import List



def is_superset(list1: List[int], list2: List[int]) -> bool:

    """Checks whether a given list is a superset of another list.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        True if the first list is a superset of the second list, False otherwise.

    """

    set1 = set(list1)

    set2 = set(list2)

    return set2.issubset(set1)

