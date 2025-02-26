from typing import List



def find_missing_elements(list1: List[int], list2: List[int]) -> List[int]:

    """Finds missing elements from the first list that are not present in the second list.



    Args:

        list1: The first list of integers.

        list2: The second list of integers.



    Returns:

        A list of missing elements from the first list that are not present in the second list.

    """

    set1 = set(list1)

    set2 = set(list2)

    missing = set1 - set2

    return list(missing)

