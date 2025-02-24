from typing import Dict, List, Any



def get_list_of_dicts(d: Dict[str, List[Any]]) -> List[Dict[str, List[Any]]]:

    """

    Returns a list of dictionaries for all items in the input dictionary where the dictionary keys are strings and the dictionary values are lists.



    Args:

        d: The input dictionary.



    Returns:

        A list of dictionaries.

    """

    list_of_dicts = []



    for key, value in d.items():

        if isinstance(key, str) and isinstance(value, list):

            list_of_dicts.append({key: value})



    return list_of_dicts

