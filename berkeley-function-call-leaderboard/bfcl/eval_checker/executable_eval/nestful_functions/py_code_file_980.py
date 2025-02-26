from typing import Dict, Any



def compose_func_map(func: callable, key_map: Dict[str, Any]) -> Dict[str, Any]:

    """Applies a function to the values of a dictionary based on a mapping of keys.



    Args:

        func: The function to be applied.

        key_map: A dictionary mapping keys to other keys.



    Returns:

        A new dictionary with the new values mapped to the original keys.

    """

    new_dict = {}

    for key, value in key_map.items():

        new_value = func(value)

        new_dict[key] = new_value

    return new_dict

