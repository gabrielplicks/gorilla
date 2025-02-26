from typing import Dict



def check_nested_dicts(input_dict: Dict) -> bool:

    """Checks if the input is a nested dictionary.



    Args:

        input_dict: The input dictionary to check.



    Returns:

        True if the input is a nested dictionary and False otherwise.

    """

    if not isinstance(input_dict, dict):

        return False



    for value in input_dict.values():

        if isinstance(value, dict):

            return check_nested_dicts(value)

    return True

