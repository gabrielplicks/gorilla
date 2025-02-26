from typing import Dict



def get_intersected_keys(dict_1: Dict, dict_2: Dict) -> set:

    """Returns a set of keys that are common to both dictionaries.



    Args:

        dict_1: The first dictionary.

        dict_2: The second dictionary.

    """

    return set(dict_1.keys()) & set(dict_2.keys())

