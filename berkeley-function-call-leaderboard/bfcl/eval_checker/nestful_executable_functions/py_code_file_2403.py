from typing import Dict, List



def find_extra_keys(a: Dict[str, any], b: Dict[str, any]) -> List[str]:

    """Finds the keys in `a` that are not in `b`.



    Args:

        a: A dictionary with unique string keys.

        b: A dictionary with unique string keys.



    Returns:

        A list of keys that are in `a` but not in `b`.

    """

    extra_keys = []



    for key in a:

        if key not in b:

            extra_keys.append(key)



    return extra_keys

