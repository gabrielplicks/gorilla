from typing import List



def set_overlap(l1: List[Any], l2: List[Any]) -> set:

    """Calculates the overlap between two lists.



    Args:

        l1: The first list.

        l2: The second list.



    Returns:

        The overlap between the two lists as a set.

    """

    if not isinstance(l1, list) or not isinstance(l2, list):

        raise TypeError("Both arguments must be lists")

    return set(l1) & set(l2)

