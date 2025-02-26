import json



def json_encode(string: str) -> str:

    """Encodes a JSON string.



    Args:

        string: The string to encode.



    Returns:

        A JSON-encoded representation of the string.

    """

    return json.dumps(string)

