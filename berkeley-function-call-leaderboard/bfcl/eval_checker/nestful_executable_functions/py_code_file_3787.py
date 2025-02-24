from typing import List



def is_unique_list(lst: List) -> bool:

    """Checks if all elements in a list are unique.



    Args:

        lst: The list to check for uniqueness.



    Returns:

        True if all elements are unique, False otherwise.

    """

    try:

        if isinstance(lst, list):

            return len(lst) == len(set(lst))

        else:

            return False

    except Exception:

        return False

