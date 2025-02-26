from typing import Dict, List



def get_keys_by_prefix(dictionary: Dict[str, any], prefix: str) -> List[str]:

    """Returns a list of keys that start with the given prefix.



    Args:

        dictionary: The dictionary to search for keys.

        prefix: The prefix to filter the keys.

    """

    return [key for key in dictionary if key.startswith(prefix)]

