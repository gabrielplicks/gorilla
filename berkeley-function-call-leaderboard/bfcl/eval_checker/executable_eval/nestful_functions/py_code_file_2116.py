from typing import Dict, Any



def merge_and_override(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:

    """Merges two dictionaries and overrides any duplicate keys with the values from the second dictionary.

    If the two dictionaries have common keys, but with different types or structures, the function throws an exception and returns None.

    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.

    """

    for key in dict2:

        if key in dict1:

            if not isinstance(dict1[key], type(dict2[key])):

                raise Exception(

                    "Cannot merge dictionaries with differing types for the same key"

                )

            else:

                dict1[key] = dict2[key]

        else:

            dict1[key] = dict2[key]

    return dict1

