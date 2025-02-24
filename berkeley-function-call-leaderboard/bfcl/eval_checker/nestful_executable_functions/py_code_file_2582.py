from typing import Dict, List



def convert_dict_list_to_list_dict(dict_list: List[Dict[str, int]]) -> Dict[str, List[int]]:

    """Converts a list of dictionaries into a dictionary of lists.



    Each list contains the values of the corresponding key from the original list of dictionaries.



    Args:

        dict_list: A list of dictionaries.



    Returns:

        A dictionary where each key is a key from the original dictionaries and each value is a list of the corresponding values from the original dictionaries.

    """

    result = {}



    for dictionary in dict_list:

        for key, value in dictionary.items():

            if key not in result:

                result[key] = []

            result[key].append(value)



    return result

