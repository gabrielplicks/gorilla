from typing import Dict, List



def find_different_keys(x: Dict[str, any], y: Dict[str, any]) -> List[str]:

    """

    Returns the list of keys that exist in both dictionaries but have different values.



    Args:

        x: The first dictionary.

        y: The second dictionary.

    """

    different_keys = []



    for key in x:

        if key in y and x[key] != y[key]:

            different_keys.append(key)



    return different_keys

