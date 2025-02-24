from typing import Dict, Any



def recursively_replace_keys(d: Dict[str, Any]) -> Dict[str, Any]:

    """Recursively replaces all keys in a given dictionary with new string values.



    The function replaces all keys in a nested dictionary with a new string value

    by concatenating the old key with '_2'.



    Args:

        d: The input dictionary.



    Returns:

        A new dictionary with all keys replaced.

    """

    if not isinstance(d, dict):

        print("Invalid input: input is not a dictionary.")

        return None



    new_dict = {}



    for key, value in d.items():

        new_key = key + '_2'

        if isinstance(value, dict):

            new_value = recursively_replace_keys(value)

        else:

            new_value = value

        new_dict[new_key] = new_value



    return new_dict

