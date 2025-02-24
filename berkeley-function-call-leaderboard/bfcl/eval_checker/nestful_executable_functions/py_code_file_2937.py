from typing import List



def append_and_return(list1: List[int], list2: List[int]) -> List[int]:

    """Appends the second list at the end of the first list and returns the first list.



    Args:

        list1: The first list.

        list2: The second list.

    """

    list1.extend(list2)

    return list1

