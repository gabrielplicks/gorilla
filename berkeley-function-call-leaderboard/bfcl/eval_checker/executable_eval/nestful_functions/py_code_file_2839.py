from typing import List



def set_difference(list_1: List[int], list_2: List[int]) -> set:

    """

    Performs set difference between two lists.

    The function returns a set that contains the elements that are in the first list but not in the second list.

    The function includes a precondition check that ensures both lists have unique elements.



    Args:

        list_1: The first list.

        list_2: The second list.



    Raises:

        ValueError: If both lists do not have unique elements.

    """

    if len(set(list_1)) != len(list_1) or len(set(list_2)) != len(list_2):

        raise ValueError("Both lists must have unique elements")

    set_1 = set(list_1)

    set_2 = set(list_2)

    result = set_1.difference(set_2)

    return set(result)

