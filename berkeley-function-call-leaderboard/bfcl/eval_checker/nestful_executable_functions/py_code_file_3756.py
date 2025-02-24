from typing import Dict, Any, List



def get_last_layer_keys(dictionary: Dict[str, Any]) -> List[str]:

    """Recursively traverses a dictionary and returns a list of keys in the last layer of the dictionary.



    Args:

        dictionary: The dictionary to traverse.



    Returns:

        A list of keys in the last layer of the dictionary.

    """

    keys = []

    for key, value in dictionary.items():

        if isinstance(value, dict):

            keys.extend(get_last_layer_keys(value))

        else:

            keys.append(key)

    return keys

