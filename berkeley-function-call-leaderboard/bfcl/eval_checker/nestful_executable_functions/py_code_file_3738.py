from typing import Dict



def merge_dicts_with_keys_in_both_dicts(dict1: Dict[str, int], dict2: Dict[str, int]) -> Dict[str, int]:

    """Merges two dictionaries into a new dictionary that only contains the key/value pairs from dict1 that are also in dict2.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        A new dictionary that only contains the key/value pairs from dict1 that are also in dict2.

    """

    merged_dict = {}

    for key in dict1.keys():

        if key in dict2:

            merged_dict[key] = dict1[key]

    return merged_dict

