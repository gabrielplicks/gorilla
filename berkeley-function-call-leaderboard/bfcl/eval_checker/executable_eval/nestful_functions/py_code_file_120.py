from typing import Dict



def check_if_matching(dict1: Dict[str, int], dict2: Dict[str, int]) -> bool:

    """Checks if the key-value pairs in each dictionary match.



    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.



    Returns:

        True if the key-value pairs in each dictionary match, False otherwise.

    """

    if len(dict1) != len(dict2):

        return False



    for key in dict1:

        if key not in dict2:

            return False

        if not isinstance(dict1[key], type(dict2[key])):

            return False

        if dict1[key] != dict2[key]:

            return False



    return True

