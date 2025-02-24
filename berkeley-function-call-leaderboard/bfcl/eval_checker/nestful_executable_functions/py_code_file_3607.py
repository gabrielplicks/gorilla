from typing import List



def list_sum(list1: List[float], list2: List[float]) -> List[float]:

    """

    Takes two lists of the same length and returns a new list containing the sum of the corresponding elements in the input lists.

    Args:

        list1: The first input list.

        list2: The second input list.

    """

    return [x + y for x, y in zip(list1, list2)]

