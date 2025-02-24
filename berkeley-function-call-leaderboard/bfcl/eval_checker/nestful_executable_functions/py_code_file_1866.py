from typing import Dict



def dictionaries_are_equivalent(dict1: Dict[str, object], dict2: Dict[str, object]) -> bool:

    """

    Checks if two dictionaries are equivalent.

    Two dictionaries are equivalent if they have the same number of keys, the same key-value pairs,

    and the values associated with each key are equal.

    Args:

        dict1: The first dictionary.

        dict2: The second dictionary.

    Returns:

        True if the dictionaries are equivalent, False otherwise.

    """

    if len(dict1) != len(dict2):

        return False

    for key in dict1:

        if key not in dict2:

            return False

        if dict1[key] != dict2[key]:

            return False

    return True

