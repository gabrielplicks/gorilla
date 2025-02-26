from typing import Dict, List, Any



def dict_to_list_of_dicts(input_dict: Dict[str, Any]) -> List[Dict[str, Any]]:

    """Converts a dictionary into a list of dictionaries, where each dictionary contains a single key-value pair from the input dictionary.



    Args:

        input_dict: The input dictionary.



    Returns:

        A list of dictionaries. Each dictionary in the output list contains a single key-value pair from the input dictionary.

    """

    output_list = []

    for key, value in input_dict.items():

        output_dict = {}

        output_dict[key] = value

        output_list.append(output_dict)

    return output_list

