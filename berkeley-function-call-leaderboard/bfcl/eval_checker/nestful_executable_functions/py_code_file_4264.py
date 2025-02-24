from typing import Dict



def dictionary_intersection(d1: Dict[str, int], d2: Dict[str, int]) -> Dict[str, int]:

    """Returns a new dictionary containing only the keys that are present in both dictionaries.

    Args:

        d1: The first dictionary.

        d2: The second dictionary.

    """

    intersection = {}

    for key in d1.keys():

        if key in d2 and d1[key] == d2[key]:

            intersection[key] = d1[key]

    return intersection

