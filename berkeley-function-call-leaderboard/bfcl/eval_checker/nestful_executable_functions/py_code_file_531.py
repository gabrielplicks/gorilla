from typing import Dict



def create_combined_dict(dict1: Dict, dict2: Dict) -> Dict:

    """Creates a new dictionary that contains all the keys and values from both `dict1` and `dict2`.

    If a key appears in both dictionaries, the value from `dict2` is used.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        A new dictionary that contains all the keys and values from both `dict1` and `dict2`.

    """

    result = {}

    for key in dict1:

        result[key] = dict1[key]

    for key in dict2:

        if key in result:

            result[key] = dict2[key]

        else:

            result[key] = dict2[key]

    return result

