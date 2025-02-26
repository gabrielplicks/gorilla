from typing import Dict



def copy_with_defaults(data: Dict[str, int], defaults: Dict[str, int]) -> Dict[str, int]:

    """Creates a copy of the data dictionary that includes all keys and values from data,

    as well as any additional keys (and their values) from defaults that are not already

    present in data. If a key exists in both data and defaults, the value in data takes

    precedence.



    Args:

        data: A dictionary containing the initial data.

        defaults: A dictionary containing default values.

    """

    new_dict = data.copy()

    for key in defaults.keys():

        if key not in new_dict:

            new_dict[key] = defaults[key]

    return new_dict

