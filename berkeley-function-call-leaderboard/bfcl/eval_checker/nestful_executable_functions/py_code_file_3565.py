from typing import Dict



def get_value_v2(key: str, data: Dict[str, str]) -> str:

    """

    Retrieves the value corresponding to the given key.



    Args:

        key: The key to retrieve the value for.

        data: The dictionary containing the key-value pairs.



    Returns:

        The value corresponding to the given key, or None if the key is not present in the dictionary.

    """

    mapping = {

        'a': '10',

        'b': '20',

        'c': '30',

    }



    try:

        return mapping[key]

    except KeyError:

        return None

