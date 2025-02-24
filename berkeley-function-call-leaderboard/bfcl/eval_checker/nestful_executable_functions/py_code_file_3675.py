from typing import Dict, List



def common_keys(dict1: Dict, dict2: Dict) -> List:

    """Returns a list of keys that are common to both dictionaries.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.

    """

    return list(set(dict1.keys()) & set(dict2.keys()))

