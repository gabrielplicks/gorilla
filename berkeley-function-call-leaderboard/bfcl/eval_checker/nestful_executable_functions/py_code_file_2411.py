from typing import Dict, List



def get_valid_dict_items(d: Dict[str, any], items: List[str]) -> List[any]:

    """Returns a list of values from the dictionary `d` associated with the keys from `items`.

    Args:

        d: A dictionary containing key-value pairs.

        items: A list of strings representing keys in the dictionary `d`.

    """

    result = []

    for item in items:

        if item in d:

            result.append(d[item])

    return result

