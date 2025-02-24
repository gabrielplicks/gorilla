from typing import List



def find_difference(list_1: List[int], list_2: List[int]) -> List[int]:

    """Finds the elements that are in the first list but not in the second.



    Args:

        list_1: The first list of integers.

        list_2: The second list of integers.



    Returns:

        A list of integers that are in the first list but not in the second.

    """

    return list(set(list_1) - set(list_2))

