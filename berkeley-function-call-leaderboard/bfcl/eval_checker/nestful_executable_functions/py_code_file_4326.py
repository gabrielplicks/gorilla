from typing import Dict, List



def consolidate_dicts(list_of_dicts: List[Dict[str, any]]) -> Dict[str, any]:

    """

    Consolidates a list of dictionaries into a single dictionary.

    If a key appears in multiple dictionaries, the value from the first dictionary in the list is used.



    Args:

        list_of_dicts: A list of dictionaries to consolidate.



    Returns:

        A dictionary containing the key-value pairs from all input dictionaries.

    """

    result = {}

    for dictionary in list_of_dicts:

        for key, value in dictionary.items():

            if key not in result:

                result[key] = value

    return result

