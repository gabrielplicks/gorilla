from typing import Dict, List, Union



def find_by_key(data: Dict[str, Union[int, float]], keys: List[str]) -> Union[int, float, None]:

    """

    Finds the first value in `data` that matches the first key in `keys`.



    Args:

        data: A dictionary containing key-value pairs.

        keys: A list of keys to search for in `data`.



    Returns:

        The first value in `data` that matches the first key in `keys`, or `None` if no match is found.

    """

    if keys is None or len(keys) == 0:

        return None



    for key in keys:

        if key in data:

            return data[key]



    return None

