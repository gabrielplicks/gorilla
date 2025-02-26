from typing import Dict, List



def get_keys_with_prefix(dictionary: Dict[str, int], prefix: str) -> List[str]:

    """Returns a list of all keys in a dictionary that start with a given prefix.



    Args:

        dictionary: The dictionary to search.

        prefix: The prefix to search for.

    """

    keys_with_prefix = []



    for key in dictionary:

        if key.startswith(prefix):

            keys_with_prefix.append(key)



    return keys_with_prefix

