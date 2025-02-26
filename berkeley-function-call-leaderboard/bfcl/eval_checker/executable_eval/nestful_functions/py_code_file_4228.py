from typing import List



def different_elements(list_1: List[int], list_2: List[int]) -> List[int]:

    """

    Returns a sorted list of all the elements that are present in the first list but not the second.



    Args:

        list_1: The first list.

        list_2: The second list.

    """

    return sorted(set(list_1) - set(list_2))

