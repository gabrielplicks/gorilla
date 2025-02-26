from typing import Dict, Any



def count_depth(nested_dict: Dict[str, Any]) -> int:

    """

    Returns the depth of a nested dictionary by traversing through the dictionary and incrementing a counter for each level.

    If the input is not a dictionary, the function returns None.

    A try-except block is used to handle the case of a key not existing in the dictionary.



    Args:

        nested_dict: The nested dictionary to traverse.

    """

    if not isinstance(nested_dict, dict):

        return None



    depth = 1

    for key, value in nested_dict.items():

        if isinstance(value, dict):

            try:

                depth = max(depth, 1 + count_depth(nested_dict[key]))

            except KeyError:

                pass



    return depth

