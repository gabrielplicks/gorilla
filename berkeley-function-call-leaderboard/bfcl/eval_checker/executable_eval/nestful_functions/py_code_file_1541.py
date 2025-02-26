from typing import List



def make_dict(keys: List[str], values: List[str]) -> dict:

    """Creates a dictionary with the first list as the keys and the second list as the values.

    Args:

        keys: The list of keys.

        values: The list of values.

    """

    result = {}

    for key, value in zip(keys, values):

        result[key] = value

    return result

