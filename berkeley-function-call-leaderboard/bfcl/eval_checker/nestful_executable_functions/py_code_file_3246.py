from typing import Dict, Any



def check_nested_dict(dict_obj: Dict[str, Any], key: str, default: bool = False) -> bool:

    """Checks whether a given dictionary contains a nested dictionary with a given key.



    Args:

        dict_obj: The dictionary to check.

        key: The key to search for.

        default: The value to return if the key is not found.

    """

    if key in dict_obj:

        return True

    for value in dict_obj.values():

        if isinstance(value, dict):

            return check_nested_dict(value, key, default)

    return default

