import json



def parse_json_default(json_string: str) -> dict:

    """Converts a JSON object string into a Python dictionary, with each key's value populated by a default value.



    Args:

        json_string: The JSON object string to be converted.



    Returns:

        A Python dictionary with the default values for each key.

    """

    parsed_dict = json.loads(json_string)

    default_dict = {}



    def convert_to_default(value):

        if isinstance(value, dict):

            return {key: convert_to_default(val) for key, val in value.items()}

        elif isinstance(value, list):

            return [convert_to_default(item) for item in value]

        else:

            return None



    for key, value in parsed_dict.items():

        default_dict[key] = convert_to_default(value)



    return default_dict

