from collections import defaultdict

from typing import Dict, List, Any



def create_nested_dict(input_list: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:

    """Creates a new nested dictionary structure based on the input list.



    The outer dictionary should have keys that are the values of the "last" key in each input dictionary,

    with the corresponding value being a list of dictionaries. The list should contain all the input

    dictionaries that have the same value for the "last" key.



    Args:

        input_list: A list of dictionaries. Each dictionary should have a "last" key.



    Returns:

        A nested dictionary structure.

    """

    nested_dict = defaultdict(list)

    for item in input_list:

        nested_dict[item['last']].append(item)

    return dict(nested_dict)

