from typing import Dict, List, Any



def create_subset_dict(dict: Dict[Any, Any], keys: List[Any]) -> Dict[Any, Any]:

    """Creates a new dictionary with a subset of keys from the original dictionary.



    Args:

        dict: The original dictionary.

        keys: The keys to include in the new dictionary.



    Returns:

        A new dictionary with the specified keys.

    """

    subset_dict = {}



    for key in keys:

        if key in dict:

            subset_dict[key] = dict[key]

        else:

            subset_dict[key] = None



    return subset_dict

