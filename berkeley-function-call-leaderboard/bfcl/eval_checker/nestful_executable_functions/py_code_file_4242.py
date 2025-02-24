import json



def validate_json_and_parse(json_string: str) -> dict:

    """Validates a JSON string and returns a dictionary of the parsed content.



    Args:

        json_string: The JSON string to validate and parse.



    Raises:

        ValueError: If the input is not a valid JSON string.

    """

    try:

        return json.loads(json_string)

    except json.JSONDecodeError as e:

        raise ValueError("Invalid JSON format") from e

