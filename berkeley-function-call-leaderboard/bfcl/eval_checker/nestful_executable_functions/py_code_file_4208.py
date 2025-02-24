from typing import Dict, List



def get_dict_from_keys(d: Dict[str, any], l: List[str]) -> Dict[str, any]:

    """

    Returns a new dictionary with only the keys corresponding to the values from the list.

    If the key is not present in the dictionary, it should not be included in the result.



    Args:

        d: The dictionary to filter.

        l: The list of keys to include in the result.

    """

    result = {}

    for key in l:

        if key in d:

            result[key] = d[key]

    return result

