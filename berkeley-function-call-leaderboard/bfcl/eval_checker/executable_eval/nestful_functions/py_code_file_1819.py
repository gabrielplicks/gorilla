from typing import Dict, Any



def find_key_index(ordered_dict: Dict[Any, Any], target_value: Any) -> int:

    """Finds the index of the key for a given value in a custom ordered dictionary.



    Args:

        ordered_dict: A custom ordered dictionary with an ordered list of keys.

        target_value: The value to search for in the dictionary.



    Returns:

        The index of the key if found, otherwise -1.

    """

    for i, key in enumerate(ordered_dict.keys()):

        if ordered_dict[key] == target_value:

            return i



    return -1

