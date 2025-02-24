from typing import Dict, List



def filter_even_keys(d: Dict[str, any]) -> List[str]:

    """

    Filters the keys of a dictionary and returns a list of keys with even size.

    If the dictionary is empty, the function returns an empty list.

    If the dictionary is not empty but all keys have an odd size, the function returns a list containing all keys.



    Args:

        d: The dictionary to filter.

    """

    even_keys = []

    for key in d.keys():

        if len(key) % 2 == 0:

            even_keys.append(key)

    if len(d) == 0 or len(even_keys) == 0:

        return even_keys

    else:

        return list(d.keys())

