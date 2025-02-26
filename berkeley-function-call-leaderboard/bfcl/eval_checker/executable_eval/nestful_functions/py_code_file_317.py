from typing import List, Union



def dict_from_keys(keys: List[str], default: Union[None, int, float] = None) -> dict:

    """Creates a dictionary with each key mapped to the provided default value.



    Args:

        keys: A list of keys.

        default: The default value for each key. If not provided, defaults to None.

    """

    return {key: default for key in keys}

