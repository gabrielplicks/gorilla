from typing import Dict, List, Union



def create_dict_with_default_value(keys: List[str], default_value: Union[str, None]) -> Dict[str, Union[str, Dict]]:

    """Creates a dictionary where each key in `keys` maps to `default_value`.

    If `default_value` is `None`, set the default value to an empty dictionary.



    Args:

        keys: A list of strings to use as keys in the dictionary.

        default_value: The value to map each key to. If `None`, use an empty dictionary.



    Returns:

        A dictionary where each key in `keys` maps to `default_value`.

    """

    if default_value is None:

        default_value = {}



    return {key: default_value for key in keys}

