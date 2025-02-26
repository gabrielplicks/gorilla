from typing import Dict, List, Any



def get_values_in_order(dictionary: Dict[str, Any], keys: List[str]) -> List[Any]:

    """

    Returns the values of the keys in the order specified by the keys array.

    If a key doesn't exist in the dictionary, returns None for that position.



    Args:

        dictionary: The dictionary object.

        keys: The array of keys.

    """

    return [dictionary.get(key) for key in keys]

