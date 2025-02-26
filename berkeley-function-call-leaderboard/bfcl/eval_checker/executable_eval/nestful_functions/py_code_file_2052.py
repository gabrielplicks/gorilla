from typing import Dict, List, Union



def get_values_from_keys(keys: str, dictionary: Dict[str, str]) -> List[Union[str, None]]:

    """

    Takes a string of comma-separated keys and returns a list of the corresponding values in the same order.

    If a key is not in the dictionary, the corresponding value is None.



    Args:

        keys: A string of comma-separated keys.

        dictionary: A dictionary of {key: name} pairs.

    """

    result = []

    for key in keys.split(','):

        if key in dictionary:

            result.append(dictionary[key])

        else:

            result.append(None)

    return result

