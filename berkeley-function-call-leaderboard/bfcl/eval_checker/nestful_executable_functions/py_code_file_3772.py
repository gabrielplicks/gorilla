from typing import Dict, List, Any



def get_keys(dictionary: Dict[int, str]) -> List[int]:

    """

    Returns a list of keys from the given dictionary that satisfy the following conditions:

    - Each key is an integer.

    - Each value is a string.

    - Each value is equal to the key converted to a string.



    Args:

        dictionary: The dictionary to extract keys from.



    Returns:

        A list of keys that satisfy the conditions.

    """

    keys = []



    for key, value in dictionary.items():

        if type(key) == int and type(value) == str and str(key) == value:

            keys.append(key)



    return keys

