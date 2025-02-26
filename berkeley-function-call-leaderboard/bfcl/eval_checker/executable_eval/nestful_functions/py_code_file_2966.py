import inspect

from typing import Dict, List, Union



def convert_dict_to_list_of_dicts(input_dict: Dict[str, Union[str, int, float, None, callable]]) -> List[Dict[str, Union[str, int, float, None]]]:

    """Converts a dictionary to a list of dictionaries containing the dictionary keys and their associated values.

    If a dictionary key has a corresponding function value, the key-value pair is skipped.

    Handles errors when the input is not a dictionary or the function value is missing.



    Args:

        input_dict: The input dictionary.



    Returns:

        A list of dictionaries containing the dictionary keys and their associated values.

    """

    if not isinstance(input_dict, dict):

        raise TypeError("Input must be a dictionary.")



    output = []



    for key, value in input_dict.items():

        if not inspect.isfunction(value):

            output.append({key: value})



    return output

