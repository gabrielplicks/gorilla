from typing import Dict, Union



def convert_dict_type(dictionary: Dict[str, Union[str, Dict[str, Union[str, int, float]]]]) -> Dict[str, Union[int, float, Dict[str, Union[int, float]]]]:

    """Recursively converts the type of a given dictionary from string to int or float.

    Args:

        dictionary: The dictionary to convert.

    Returns:

        A new dictionary with converted types.

    """

    result = {}

    for key, value in dictionary.items():

        if isinstance(value, dict):

            result[key] = convert_dict_type(value)

        elif isinstance(value, str) and value.isdigit():

            result[key] = int(value)

        elif isinstance(value, str) and value.replace('.', '', 1).isdigit():

            result[key] = float(value)

        else:

            result[key] = value

    return result

