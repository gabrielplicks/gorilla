from typing import List



def mirror_list(list_a: List[Any]) -> List[Any]:

    """

    Returns a new list with the same elements in the same order, but mirrored instead of being ordered.

    For example, given the list [1, 2, 3, 4], the function should return [4, 3, 2, 1].

    Args:

        list_a: the list to be mirrored.

    Returns:

        a new list with the mirrored elements.

    """

    return [element for element in reversed(list_a)]

