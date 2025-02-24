from typing import Dict, List



def keep_keys(d: Dict[str, int], keys: List[str]) -> Dict[str, int]:

    """Creates a new dictionary containing only the key-value pairs from the original dictionary that have keys in the given list.



    Args:

        d: The original dictionary.

        keys: The list of keys to keep in the new dictionary.

    """

    return {key: value for key, value in d.items() if key in keys}

