from typing import Dict, List



def prefix_match(dictionary: Dict[str, int], prefix: str) -> List[str]:

    """

    Returns a list of keys from the dictionary that have the given string as a prefix.

    Additionally, the function removes any keys from the dictionary that are not present in the output list.



    Args:

        dictionary: The dictionary to search for keys with the given prefix.

        prefix: The prefix to search for in the keys of the dictionary.

    """

    filtered_keys = [key for key in dictionary if key.startswith(prefix)]

    for key in dictionary.keys():

        if key not in filtered_keys:

            del dictionary[key]



    return filtered_keys

