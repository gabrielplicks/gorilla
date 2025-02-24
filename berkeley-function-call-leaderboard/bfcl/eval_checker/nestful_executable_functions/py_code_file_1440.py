from typing import List, Dict



def add_id_to_dicts(dicts: List[Dict[str, str]]) -> List[Dict[str, str]]:

    """Adds a new key named `id` to each dictionary in a list of dictionaries.



    The `id` values are incremented starting from 1 for each dictionary in the list.



    Args:

        dicts: A list of dictionaries.



    Returns:

        A list of dictionaries with the added `id` keys.

    """

    return [{'id': i + 1, **d} for i, d in enumerate(dicts)]

