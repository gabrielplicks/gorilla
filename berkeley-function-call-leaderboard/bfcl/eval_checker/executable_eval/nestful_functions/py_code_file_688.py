from typing import List



def has_consecutive_ones(lst: List[int]) -> bool:

    """Checks if a list of integers contains a consecutive pair of 1's.



    Args:

        lst: A list of integers.



    Returns:

        True if the list contains a consecutive pair of 1's, False otherwise.

    """

    if len(lst) <= 1:

        return False

    for i in range(len(lst) - 1):

        if lst[i] == 1 and lst[i + 1] == 1:

            return True

    return False

