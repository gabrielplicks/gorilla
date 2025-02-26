from typing import Dict, List, Tuple



def get_sorted_pairs(dictionary: Dict[str, int]) -> List[Tuple[str, int]]:

    """

    Returns a sorted list of key-value pairs based on the values of the dictionary.

    The pairs are sorted in descending order based on the values.

    Args:

        dictionary: The input dictionary.

    """

    return sorted(dictionary.items(), key=lambda pair: pair[1], reverse=True)

