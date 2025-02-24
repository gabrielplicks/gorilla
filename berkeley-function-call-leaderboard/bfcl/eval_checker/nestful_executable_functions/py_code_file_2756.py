from typing import List, Union



def all_same(lst: List[Union[str, int, float]]) -> bool:

    """Checks if all elements in the list are the same.

    Args:

        lst: The input list.

    """

    elem_set = set()

    for elem in lst:

        elem_set.add(elem)

    return len(elem_set) == 1



def all_same_with_type_check(lst: List[Union[str, int, float]]) -> bool:

    """Checks if all elements in the list are the same and of string type.

    Args:

        lst: The input list.

    """

    if not lst:

        return True

    elem_set = set()

    for elem in lst:

        if not isinstance(elem, str):

            return False

        elem_set.add(elem)

    return len(elem_set) == 1

