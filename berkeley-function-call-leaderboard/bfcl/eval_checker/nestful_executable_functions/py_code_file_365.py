from typing import List



def match_lists(list1: List[int], list2: List[int]) -> List[int]:

    """

    Compares each element in `list1` to its corresponding element in `list2` and returns a list of integers based on a specific condition.



    Args:

        list1: A list of integers.

        list2: A list of integers.



    Returns:

        A list of integers.

    """

    result = []

    for i in range(len(list1)):

        if list1[i] > list2[i]:

            result.append(1)

        elif list1[i] < list2[i]:

            result.append(2)

        else:

            result.append(3)

    return result

