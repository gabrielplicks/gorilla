from typing import Dict, Tuple



def get_sorted_key_value_pairs_by_value(dictionary: Dict[str, int], ascending: bool = True) -> List[Tuple[str, int]]:

    """

    Returns a list of key-value pairs sorted by value in either ascending or descending order.



    Args:

        dictionary: A dictionary of key-value pairs.

        ascending: A boolean indicating whether the order should be ascending (True) or descending (False).



    Returns:

        A list of key-value pairs sorted by value.

    """

    sorted_items = sorted(dictionary.items(), key=lambda x: x[1])

    if not ascending:

        sorted_items = sorted_items[::-1]

    return sorted_items

