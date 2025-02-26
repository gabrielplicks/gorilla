from typing import Dict



def merge_dicts_with_precedence(dict1: Dict, dict2: Dict) -> Dict:

    """Merges two dictionaries, where the values in the second dictionary take precedence over the values in the first dictionary if the same key is present.

    A shallow copy of the first dictionary is used, so the original first dictionary is not modified.

    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.

    """

    merged_dict = dict1.copy()



    for key, value in dict2.items():

        merged_dict[key] = value



    return merged_dict

