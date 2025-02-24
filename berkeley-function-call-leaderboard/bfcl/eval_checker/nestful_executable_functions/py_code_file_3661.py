from typing import Dict, List, Any



def get_ids_by_value(dictionary: Dict[Any, Any], value: Any) -> List[Any]:

    """

    Returns a list of keys that have the given value in the dictionary.

    If no keys with the value are found, returns an empty list.



    Args:

        dictionary: The dictionary to search in.

        value: The value to search for.

    """

    ids = []

    for key, val in dictionary.items():

        if val == value:

            ids.append(key)

    return ids

