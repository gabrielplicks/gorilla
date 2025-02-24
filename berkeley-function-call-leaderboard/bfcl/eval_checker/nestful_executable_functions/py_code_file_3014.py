import json



def parse_json_str(json_str: str) -> dict:

    """Parses a JSON string and returns the corresponding Python data structure.



    Args:

        json_str: The JSON string to parse.



    Raises:

        ValueError: If the input is not a valid JSON string.

    """

    try:

        return json.loads(json_str)

    except json.JSONDecodeError:

        raise ValueError('Invalid JSON string')

