from typing import List



def sublists_of_size5(lst: List[str]) -> List[List[str]]:

    """Creates a list of all sub-lists of size 5 that can be formed from the given list.



    Args:

        lst: The input list.



    Returns:

        A list of sub-lists of size 5.

    """

    result = []

    for i in range(len(lst) - 4):

        sublist = lst[i:i+5]

        result.append(sublist)

    return result

