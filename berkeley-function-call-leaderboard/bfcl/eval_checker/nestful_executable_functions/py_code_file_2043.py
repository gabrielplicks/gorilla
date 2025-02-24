from typing import List



def find_longest_common_sublist(list1: List[int], list2: List[int]) -> List[int]:

    """Finds the longest common sublist between two lists.



    Args:

        list1: A list of integers.

        list2: A list of integers.



    Returns:

        A list containing the longest common sublist between the two lists.

    """

    if len(list1) == 0 or len(list2) == 0:

        return []

    if list1[0] == list2[0]:

        return [list1[0]] + find_longest_common_sublist(list1[1:], list2[1:])

    if len(list1) > len(list2):

        return find_longest_common_sublist(list1[1:], list2)

    else:

        return find_longest_common_sublist(list1, list2[1:])

