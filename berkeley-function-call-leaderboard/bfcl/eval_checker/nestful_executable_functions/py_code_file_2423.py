from typing import List



def initialize_list_to_none(l: List[Any]) -> List[None]:

    """Creates a new list with all entries initialized to None.



    Args:

        l: The input list.



    Returns:

        A new list with all entries initialized to None.

    """

    return [None for _ in l.copy()]

