from typing import Dict, Any



def truncate_dict_values(data: Dict[str, Any], max_length: int) -> Dict[str, Any]:

    """Truncates any values in a nested dictionary that are longer than a specified maximum length.



    Args:

        data: The nested dictionary to be truncated.

        max_length: The maximum length of the values.

    """

    truncated_data = {}



    for key, value in data.items():

        if isinstance(value, dict):

            truncated_data[key] = truncate_dict_values(value, max_length)

        elif isinstance(value, str) and len(value) > max_length:

            truncated_data[key] = value[:max_length] + '...'

        else:

            truncated_data[key] = value



    return truncated_data

