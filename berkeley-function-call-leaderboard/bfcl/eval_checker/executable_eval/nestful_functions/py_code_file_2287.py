from typing import Dict, Any



def compare_keys(dict1: Dict[Any, Any] = None, dict2: Dict[Any, Any] = None) -> bool:

    """

    Compares the keys of two dictionaries and returns True if the keys in dict1 are a subset of the keys in dict2.

    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.

    """

    dict1 = dict1 or {}

    dict2 = dict2 or {}

    if not isinstance(dict1, dict) or not isinstance(dict2, dict):

        raise TypeError("Both `dict1` and `dict2` should be dictionaries")

    return set(dict1.keys()).issubset(dict2.keys())

