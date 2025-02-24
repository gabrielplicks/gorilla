from typing import Dict, List, Any



def access_attrs(json: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:

    """

    Returns a dictionary with the same keys and their corresponding values from the input dictionary.

    If a key does not exist in the input dictionary, it is added to the result dictionary with a None value.

    Args:

        json: The input dictionary.

        keys: A list of keys to access in the input dictionary.

    Returns:

        A dictionary with the same keys and their corresponding values from the input dictionary.

    """

    result = {}

    for key in keys:

        result[key] = json.get(key)

    return result

