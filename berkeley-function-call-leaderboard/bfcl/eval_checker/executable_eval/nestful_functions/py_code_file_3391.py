import copy

from typing import Dict, List



def deep_copy_matching_keys(dictionary: Dict, keys: List[str]) -> Dict:

    """Creates a deep copy of a dictionary that only contains the keys that match the given keys.



    Args:

        dictionary: The dictionary to copy.

        keys: The keys to keep in the new dictionary.



    Returns:

        A deep copy of the dictionary that only contains the matching keys.

    """

    new_dictionary = copy.deepcopy(dictionary)

    for key in list(new_dictionary.keys()):

        if key not in keys:

            del new_dictionary[key]

    return new_dictionary

