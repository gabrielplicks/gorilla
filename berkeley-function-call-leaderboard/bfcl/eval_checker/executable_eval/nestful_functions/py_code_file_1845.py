from typing import List



def all_zero(lst: List[int]) -> bool:

    """Checks if all elements in a list are zero.



    Args:

        lst: The list to check.



    Returns:

        A boolean value indicating whether all elements in the list are zero.

    """

    return all(x == 0 for x in lst)

