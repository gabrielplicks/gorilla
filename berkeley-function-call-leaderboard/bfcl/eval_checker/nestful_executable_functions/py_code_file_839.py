import json

from typing import Any, Dict



def json_to_nested_dict(json_string: str) -> Dict[str, Any]:

    """Converts a JSON string into a nested dictionary.



    Args:

        json_string: The JSON string to be converted.



    Returns:

        A nested dictionary or an error message if the JSON string is malformed.

    """

    try:

        data = json.loads(json_string)

    except json.decoder.JSONDecodeError:

        return "Invalid JSON string"



    def convert(d: Dict[str, Any]) -> Dict[str, Any]:

        if isinstance(d, dict):

            return {k: convert(v) for k, v in d.items()}

        return d



    return convert(data)

