from typing import Dict



def sum_dictionary_of_dictionaries(d: Dict[str, Dict[str, int]]) -> Dict[str, int]:

    """Sum the values of a dictionary of dictionaries.



    Args:

        d: A dictionary of dictionaries.



    Returns:

        A dictionary with the sum of the values for each key.

    """

    result = {}

    for key, inner_dict in d.items():

        for inner_key, value in inner_dict.items():

            if inner_key in result:

                result[inner_key] += value

            else:

                result[inner_key] = value

    return result

