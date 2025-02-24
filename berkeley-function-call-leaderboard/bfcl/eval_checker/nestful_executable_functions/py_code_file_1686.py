from typing import Dict, Any



def swap_sum(d: Dict[str, Dict[str, Any]]) -> Dict[str, int]:

    """Swaps keys and values in nested dictionaries and sums the values for each key.



    Args:

        d: A dictionary containing nested dictionaries. Each nested dictionary contains a list of integers and strings.



    Returns:

        A new dictionary that contains the sum of values for each key.

    """

    result = {}

    for nested_dict in d.values():

        for key, value in nested_dict.items():

            if key not in result:

                result[key] = []

            result[key].append(value)

    for key in result:

        result[key] = sum(result[key])

    return result

