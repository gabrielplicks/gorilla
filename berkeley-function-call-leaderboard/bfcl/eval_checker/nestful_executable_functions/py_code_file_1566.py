from typing import Dict, Any



def convert_dict_values_to_bool(value: Dict[str, Any], case_sensitive: bool = True) -> Dict[str, bool]:

    """

    Converts a dictionary with string values to a dictionary with boolean values.



    Args:

        value: The dictionary to convert.

        case_sensitive: Whether the conversion should be case-sensitive.



    Returns:

        A new dictionary with boolean values.

    """

    if not isinstance(value, dict):

        return None

    new_dict = {}

    for key, value in value.items():

        if isinstance(value, str):

            if case_sensitive:

                value = True if value.lower() in ('true', '1') else False

            else:

                value = True if value in ('true', '1') else False

        new_dict[key] = value

    return dict(new_dict)

