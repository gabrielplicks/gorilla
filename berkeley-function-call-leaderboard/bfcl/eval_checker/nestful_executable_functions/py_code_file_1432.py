import json

import base64



def serialize_json_and_base64_encode(data: dict) -> str:

    """Converts a JSON serializable object into its Base64 encoded string.



    Args:

        data: The JSON serializable object to be converted.



    Returns:

        The Base64 encoded string of the input data.

    """

    json_data = json.dumps(data)

    base64_data = base64.b64encode(json_data.encode('utf-8'))

    return base64_data

