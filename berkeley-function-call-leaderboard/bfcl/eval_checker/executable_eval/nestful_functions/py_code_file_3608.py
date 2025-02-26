from typing import List, Union



def L2_to_A(L2: Union[List[Union[int, float, str, None]], None]) -> List[Union[int, float, str, None]]:

    """

    Returns a list containing the first two elements of the input list if it has length of at least 2,

    or the first element if it has length of 1. In case the input list is empty or contains only `None`s,

    the function should return an empty list.



    Args:

        L2: The input list.

    """

    if not isinstance(L2, list):

        return []

    if len(L2) == 0:

        return []

    if all(element is None for element in L2):

        return []

    if len(L2) <= 2:

        return L2

    return L2[:2]

