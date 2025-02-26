from typing import Dict, List, Any



def filter_dict_items(dictionary: Dict[str, Dict[str, Any]], condition: Dict[str, Any]) -> List[Dict[str, Any]]:

    """

    Filters a dictionary based on a matching condition.



    Args:

        dictionary: The dictionary to filter.

        condition: The condition to match.



    Returns:

        A list of matching values.

    """

    result = []

    for key, value in dictionary.items():

        if all(key in value and value[key] == condition[key] for key in condition):

            result.append(value)

    return result

