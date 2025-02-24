from typing import List



def check_lists(list1: List[Any], list2: List[Any]) -> bool:

    """Checks whether two given lists are equal and returns the result.

    If either list is None, returns False immediately.

    Args:

        list1: The first list to compare.

        list2: The second list to compare.

    """

    if not list1 or not list2:

        return False



    if len(list1) != len(list2):

        return False



    for i in range(len(list1)):

        if list1[i] != list2[i]:

            return False



    return True

