from typing import Dict, List



def convert_dict_of_lists_to_list_of_dicts(dict_of_lists: Dict[str, List[str]]) -> List[Dict[str, List[str]]]:

    """Converts a dict of lists to a list of dicts.



    Each converted dict contains a single key-value pair with the value being a list of 2 elements.



    Args:

        dict_of_lists: The input dict of lists.



    Returns:

        A list of dicts.

    """

    converted_list = []

    for key, value in dict_of_lists.items():

        converted_list.append({key: value})

    return converted_list

