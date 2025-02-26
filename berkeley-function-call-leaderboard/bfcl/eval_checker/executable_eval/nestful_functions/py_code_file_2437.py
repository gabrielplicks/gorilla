from typing import Dict, List, Union



def build_list_from_dicts(*dicts: Dict[str, Union[int, float]]) -> List[List[Union[int, float]]]:

    """Builds a list of lists from a variable number of dictionaries.



    Each list in the returned list is a dictionary's positional arguments.

    If the dictionary has no positional arguments, then an empty list is returned.



    Args:

        dicts: A variable number of dictionaries.



    Returns:

        A list of lists where each list is a dictionary's positional arguments.

        If the dictionary has no positional arguments, then an empty list is returned.

    """

    return [

        list(dict.values()) if dict else []

        for dict in dicts

    ]

