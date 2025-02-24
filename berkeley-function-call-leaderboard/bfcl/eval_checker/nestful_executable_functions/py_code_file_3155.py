from typing import List



def remove_duplicates_using_set(list1: List[int], list2: List[int]) -> List[int]:

    """

    Removes the elements in `list1` that are also present in `list2` using a set.



    Args:

        list1: A list of integers.

        list2: A list of integers.



    Returns:

        A list of integers that are present in `list1` but not in `list2`.

    """

    set2 = set(list2)

    result = [element for element in list1 if element not in set2]



    return result

