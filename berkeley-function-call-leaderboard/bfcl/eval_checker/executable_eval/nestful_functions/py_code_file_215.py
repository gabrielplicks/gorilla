from typing import Dict



def convert_dict_values_to_lists(dict_obj: Dict[str, str]) -> Dict[str, list]:

    """Converts the values of a dictionary into lists.



    Args:

        dict_obj: The dictionary object.



    Returns:

        A shallow copy of the dictionary with its values converted to lists.

    """

    return {key: list(value) for key, value in dict_obj.items()}

