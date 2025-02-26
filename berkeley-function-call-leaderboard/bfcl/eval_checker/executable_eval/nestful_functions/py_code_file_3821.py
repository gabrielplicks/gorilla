import base64

import json



def parse_encoded_json(encoded_string: str) -> dict:

    """Parses a base64-encoded JSON string and returns its value as a Python object.



    Args:

        encoded_string: The base64-encoded JSON string.

    """

    decoded_string = base64.b64decode(encoded_string)

    json_data = json.loads(decoded_string)

    return json_data

