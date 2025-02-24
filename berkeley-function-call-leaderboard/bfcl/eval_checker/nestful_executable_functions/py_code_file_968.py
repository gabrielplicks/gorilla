from typing import Dict



def update_dict_recursive(d1: Dict, d2: Dict) -> None:

    """Updates the dictionary `d1` with the contents of `d2` recursively.

    If a key exists in both dictionaries, the value in `d1` is updated with the value in `d2`.

    If a key exists in only one of the dictionaries, its value is added to `d1`.

    The function takes care of nested dictionaries.

    Args:

        d1: The dictionary to be updated.

        d2: The dictionary containing the updates.

    """

    for key, value in d2.items():

        if key in d1:

            if isinstance(value, dict) and isinstance(d1[key], dict):

                update_dict_recursive(d1[key], value)

            else:

                d1[key] = value

        else:

            d1[key] = value

