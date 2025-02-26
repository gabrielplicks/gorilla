from typing import List, Optional



def first_common_element(list1: List[int], list2: List[int]) -> Optional[int]:

    """

    Returns the first common element from two lists.

    If there is no common element, returns None.



    Args:

        list1: The first list.

        list2: The second list.

    """

    set1 = set(list1)

    set2 = set(list2)

    intersection = set1.intersection(set2)

    if len(intersection) == 0:

        return None

    return intersection.pop()

