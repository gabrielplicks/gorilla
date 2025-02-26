from typing import List, Dict



def create_dict_from_lists(keys: List[str], values: List[str]) -> Dict[str, str]:

    """Creates a dictionary from two lists, keys and values.



    If the number of elements in keys is less than in values, raises an exception.

    If the number of elements is greater, ignores the extra elements of values.

    If keys and values are empty, returns an empty dictionary.



    Args:

        keys: A list of keys.

        values: A list of values.



    Raises:

        Exception: If the number of keys is less than the number of values.

    """

    dictionary = {}

    if not keys or not values:

        return dictionary

    if len(keys) < len(values):

        raise Exception("Number of keys is less than the number of values")

    for i in range(min(len(keys), len(values))):

        dictionary[keys[i]] = values[i]



    return dictionary

