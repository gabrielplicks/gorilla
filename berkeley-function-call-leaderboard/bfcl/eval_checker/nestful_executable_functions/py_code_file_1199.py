from typing import Dict



def are_dicts_equal(dict1: Dict, dict2: Dict) -> tuple:

    """

    Checks if two dictionaries are equal, including nested dictionaries.



    Args:

        dict1: The first dictionary to compare.

        dict2: The second dictionary to compare.



    Returns:

        A tuple containing a boolean indicating if the dictionaries are equal and a message.

    """

    if not isinstance(dict1, dict) or not isinstance(dict2, dict):

        return False, "One of the inputs is not a dictionary."

    for key in dict1:

        if key not in dict2:

            return False, f"Key {key} not found in the other dictionary."

        if dict1[key] != dict2[key]:

            return False, f"Values for key {key} are not equal."

    return True, "The two dictionaries are equal."

