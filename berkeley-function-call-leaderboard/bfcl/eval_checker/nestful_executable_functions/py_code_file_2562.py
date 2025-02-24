from typing import Dict



def get_value_or_zero(dictionary: Dict[str, int], key: str) -> int:

    """Returns the value associated with the key if it exists, otherwise returns 0.



    Args:

        dictionary: The dictionary to search.

        key: The key to search for.



    Returns:

        The value associated with the key if it exists, otherwise 0.

    """

    return dictionary.get(key, 0)

