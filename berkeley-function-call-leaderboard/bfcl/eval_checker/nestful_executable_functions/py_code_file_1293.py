from typing import List



def remove_first(l: List[int]) -> List[int]:

    """Removes the first element from a list of integers.



    Args:

        l: The list of integers.



    Returns:

        The list with the first element removed.

    """

    if len(l) <= 1:

        return []

    return l[1:]

