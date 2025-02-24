from typing import Dict, List, Union



def update_dict_recursively(orig_dict: Dict, update_dict: Dict) -> Dict:

    """Updates a dictionary recursively and returns the updated dictionary.



    Args:

        orig_dict: The original dictionary to be updated.

        update_dict: The dictionary containing keys and values to update the original dictionary.

    """

    for key, value in update_dict.items():

        if isinstance(value, dict):

            if key not in orig_dict:

                orig_dict[key] = {}

            update_dict_recursively(orig_dict[key], value)

        elif isinstance(value, list):

            if key in orig_dict:

                orig_dict[key].extend(value)

            else:

                orig_dict[key] = value

        else:

            orig_dict[key] = value

    return orig_dict

