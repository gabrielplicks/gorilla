from typing import Dict, Any, Union



def search_nested_dict(dictionary: Dict[str, Any], key: str) -> Union[Tuple[str, Any], None]:

    """Searches for a key in a nested dictionary structure.

    Args:

        dictionary: The nested dictionary to search in.

        key: The key to search for.

    Returns:

        A tuple containing the key and its corresponding value if found, otherwise None.

    """

    if key in dictionary:

        return (key, dictionary[key])

    for value in dictionary.values():

        if isinstance(value, dict):

            result = search_nested_dict(value, key)

            if result:

                return result

    return None

