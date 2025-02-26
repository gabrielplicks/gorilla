from typing import Dict



def prefix_dict_keys(d: Dict[str, any], prefix: str) -> Dict[str, any]:

    """Prefixes all keys in a dictionary with a given string.



    Args:

        d: The dictionary to prefix the keys of.

        prefix: The string to prefix the keys with.



    Returns:

        A new dictionary with all keys prefixed by `prefix`.

    """

    return {prefix + k: d[k] for k in d}

