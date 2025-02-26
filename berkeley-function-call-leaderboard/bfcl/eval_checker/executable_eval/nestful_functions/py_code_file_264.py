import json

from typing import Any, Dict, List, Union



def get_nested_value(data: Dict[str, Any], keys: List[str]) -> Union[Any, None]:

    """

    Returns the value of a nested dictionary by specifying the desired keys in a list.

    Args:

        data: The current dictionary.

        keys: A list of keys to look for.

    """

    if len(keys) == 0:

        return data



    key = keys[0]

    if key not in data:

        return None



    return get_nested_value(data[key], keys[1:])

