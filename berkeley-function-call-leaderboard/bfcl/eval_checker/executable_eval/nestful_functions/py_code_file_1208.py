from typing import Dict



def process_two_dictionaries(dict1: Dict, dict2: Dict) -> Dict:

    """Merges two dictionaries and returns a new dictionary with all the keys from both dictionaries and their corresponding values.

    If any key exists in both dictionaries, the value from the second dictionary should be used.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        A new dictionary with all the keys from both dictionaries and their corresponding values.

    """

    result = dict1.copy()

    result.update(dict2)

    for key in dict1.keys():

        if key not in dict2:

            result.setdefault(key)

    return result

