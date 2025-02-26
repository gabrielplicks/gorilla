import json



def validate_json_dict(json_string: str) -> bool:

    """Validates a JSON string as a valid dictionary.



    Args:

        json_string: The JSON string to validate.



    Returns:

        True if the JSON string is a valid dictionary, False otherwise.

    """

    try:

        parsed_object = json.loads(json_string)

        return isinstance(parsed_object, dict)

    except json.JSONDecodeError:

        return False

