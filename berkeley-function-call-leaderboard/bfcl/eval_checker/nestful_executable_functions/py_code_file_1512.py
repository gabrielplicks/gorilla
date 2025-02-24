from typing import List



def find_integers(list1: List[int], list2: List[int]) -> List[int]:

    """Finds integers that are in the first list but not in the second list.

    Args:

        list1: The first list of integers.

        list2: The second list of integers.

    """

    set1 = set(list1)

    set2 = set(list2)

    return list(set1 - set2)

