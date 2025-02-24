from typing import Dict, List, Tuple



def case_insensitive_key_value_pairs(d: Dict[str, str]) -> List[Tuple[str, str]]:

    """

    Returns a list of all pairs of key-value pairs in the dictionary that match in both key and value, ignoring case.



    Args:

        d: The input dictionary.

    """

    pairs = []

    for key, value in d.items():

        if key.lower() == value.lower():

            pairs.append((key, value))

    return pairs

