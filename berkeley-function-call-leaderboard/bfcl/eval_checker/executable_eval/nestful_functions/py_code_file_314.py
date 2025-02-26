from typing import List, Dict



def unique_dicts(dicts: List[Dict]) -> List[Dict]:

    """Returns a list of unique dictionaries from a given list of dictionaries.



    Two dictionaries are considered equal if they have the same keys and values.



    Args:

        dicts: A list of dictionaries.



    Returns:

        A list of unique dictionaries.

    """

    unique = []

    for d in dicts:

        if d not in unique:

            unique.append(d)

    return unique

