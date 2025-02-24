from typing import List



def unique_common_values(list1: List[int], list2: List[int]) -> List[int]:

    """Returns a list of unique common values that occur in both input lists.



    Args:

        list1: The first input list.

        list2: The second input list.



    Returns:

        A list of unique common values.

    """

    return list(set(list1).intersection(set(list2)))

