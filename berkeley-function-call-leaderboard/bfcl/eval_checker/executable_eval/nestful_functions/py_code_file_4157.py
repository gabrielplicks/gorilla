from typing import List



def find_in_list(lst: List[object], element: object) -> int:

    """Finds the index of an element in a list iteratively.

    Args:

        lst: The list to search in.

        element: The element to find.

    """

    for index, item in enumerate(lst):

        if item == element:

            return index

    return -1

