from typing import List, Tuple



def search_min_max(a: List[float]) -> Tuple[float, float]:

    """Finds the minimum and maximum values in a list without modifying the original list.

    Args:

        a: A list of numeric values.

    Returns:

        A tuple containing the minimum and maximum values in the list.

    """

    return min(a), max(a)

