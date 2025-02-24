from typing import Dict, Tuple, List



def match_dict_values(dict1: Dict[str, str], dict2: Dict[str, str]) -> List[Tuple[str, str]]:

    """Finds the matching key-value pairs between two dictionaries.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        A list of matching key-value pairs.

    """

    matched_pairs = []

    for key, value in dict1.items():

        if key in dict2 and dict2[key] == value:

            matched_pairs.append((key, value))

    return matched_pairs

