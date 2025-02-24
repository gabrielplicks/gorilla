from typing import Dict



def filter_nested_dict(data: Dict, name: str, value: str) -> Dict:

    """

    Filters a nested dictionary based on specific criteria.



    Args:

        data: The nested dictionary to be filtered.

        name: The key within the nested dictionary.

        value: The corresponding value to be compared against.



    Returns:

        A filtered dictionary that only includes the matching records.

    """

    filtered_data = {}



    def recursive_filter(data: Dict, name: str, value: str):

        for key, val in data.items():

            if isinstance(val, dict):

                recursive_filter(val, name, value)

            elif key == name and val == value:

                filtered_data[key] = val



    recursive_filter(data, name, value)

    return filtered_data

