from typing import Dict, Tuple



def get_matching_pairs(d1: Dict[str, int], d2: Dict[str, int]) -> List[Tuple[str, int]]:

    """

    Returns a list of tuples of all key-value pairs in d2 that match the keys in d1.

    If a matching pair has the same value in both dictionaries, return only one copy.



    Args:

        d1: A dictionary with string keys and integer values.

        d2: A dictionary with string keys and integer values.



    Returns:

        A list of tuples, where each tuple contains a key-value pair from d2 that matches a key in d1.

    """

    matching_pairs = []

    for key, value in d1.items():

        if key in d2:

            if d2[key] == value:

                matching_pairs.append((key, value))

    return matching_pairs

