from typing import List, Union



def convert_str_to_dict(str_list: List[str], default_value: Union[str, int, float] = None) -> dict:

    """Converts a list of strings into a dictionary.

    Args:

        str_list: A list of strings.

        default_value: An optional default value to use when the key-value pair in the string is missing.

    """

    result = {}

    for s in str_list:

        parts = s.split(':')

        if len(parts) < 2:

            key = parts[0]

            value = default_value

        else:

            key, value = parts

        result[key] = value

    return result

