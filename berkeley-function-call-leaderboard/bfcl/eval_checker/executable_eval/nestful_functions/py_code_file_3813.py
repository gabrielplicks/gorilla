from typing import List



def find_common_elements(list1: List[int], list2: List[int]) -> List[int]:

    """Finds the common elements between two lists, removes duplicates, and sorts the resulting list in ascending order.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        A sorted list of unique common elements.

    """

    common_elements = list(set(list1).intersection(list2))

    unique_elements = list(set(common_elements))

    sorted_elements = sorted(unique_elements)

    return sorted_elements

