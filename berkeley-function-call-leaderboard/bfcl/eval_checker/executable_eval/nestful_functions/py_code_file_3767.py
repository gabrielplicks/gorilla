from typing import List



def check_positive_and_unique(lst: List[int]) -> bool:

    """Checks if a given list contains only positive integers and no duplicates.



    Args:

        lst: The list to be checked.



    Returns:

        True if the list meets the criteria, False otherwise.

    """

    if not all(isinstance(x, int) and x > 0 for x in lst):

        return False

    unique_elements = set(lst)

    if len(lst) == len(unique_elements):

        return True

    else:

        return False

