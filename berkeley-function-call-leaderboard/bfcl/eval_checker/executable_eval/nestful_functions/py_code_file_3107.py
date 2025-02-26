from typing import Dict, List



def get_keys_from_dict(d: Dict[str, any], keys: List[str]) -> Dict[str, any]:

    """

    Returns a new dictionary with only those keys from the original dictionary that exist in the list of keys.

    If any of the keys are not found in the original dictionary, the function returns an empty dictionary.



    Args:

        d: The original dictionary.

        keys: The list of keys to extract from the original dictionary.

    """

    new_dict = {}

    for key in keys:

        if key in d:

            new_dict[key] = d[key]

    return new_dict

