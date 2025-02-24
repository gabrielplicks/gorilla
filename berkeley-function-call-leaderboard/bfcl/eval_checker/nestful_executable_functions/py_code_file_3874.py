from typing import List



def find_first(list_a: List[int], list_b: List[int]) -> int:

    """Finds the first element in the intersection of two lists.



    Args:

        list_a: The first list.

        list_b: The second list.



    Returns:

        The first element in the intersection of the two lists, or -1 if the intersection is empty.

    """

    set_a = set(list_a)

    set_b = set(list_b)

    intersection = set_a.intersection(set_b)

    if not intersection:

        return -1

    return intersection.pop()

