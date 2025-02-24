from typing import List, Tuple



def sort_by_priority(items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:

    """Sorts a list of tuples by the priority score.



    Args:

        items: A list of tuples, each containing a string and a priority score.



    Returns:

        The sorted list of tuples.

    """

    return sorted(items, key=get_priority)



def get_priority(item: Tuple[str, int]) -> int:

    """Returns the priority score from the tuple item.



    Args:

        item: A tuple containing a string and a priority score.



    Returns:

        The priority score.

    """

    return item[1]

