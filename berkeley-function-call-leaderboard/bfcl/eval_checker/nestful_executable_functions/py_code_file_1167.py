from typing import Dict, Any



def get_first_key(data: Dict[str, Any]) -> str:

    """

    Returns the first key at the first level of a dictionary.



    Args:

        data: The dictionary to search for the first key.



    Returns:

        The first key at the first level of the dictionary as a string.

    """

    if not data:

        return None



    keys = list(data.keys())

    first_key = keys[0]



    if isinstance(data[first_key], dict):

        return get_first_key(data[first_key])



    return first_key

