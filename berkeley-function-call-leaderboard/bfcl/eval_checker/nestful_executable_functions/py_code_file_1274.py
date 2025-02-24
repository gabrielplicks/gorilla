from typing import Dict, List



def get_common_keys(dict1: Dict[str, any], dict2: Dict[str, any]) -> List[str]:

    """

    Returns a list of keys that are common to both dictionaries (dict1 and dict2).



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        A list of keys that are common to both dictionaries.

    """

    dict1_keys = dict1.keys()

    dict2_keys = dict2.keys()

    common_keys = set(dict1_keys).intersection(set(dict2_keys))

    return list(common_keys)

