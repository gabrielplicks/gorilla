import json



def convert_dict_to_json_str(dictionary: dict) -> str:

    """Converts a given dictionary to a JSON-like string.



    Args:

        dictionary: The dictionary to be converted.



    Returns:

        A JSON-like string representation of the given dictionary.

    """

    json_str = '{'

    for key, value in dictionary.items():

        if isinstance(value, dict):

            json_str += f'"{key}": {convert_dict_to_json_str(value)}, '

        elif isinstance(value, list):

            json_str += f'"{key}": {json.dumps(value)}, '

        elif value is not None:

            json_str += f'"{key}": {json.dumps(value)}, '

    if len(json_str) > 1:

        json_str = json_str[:-2]

    json_str += '}'

    return json_str

