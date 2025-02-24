import json



def get_status_value(json_str: str) -> str:

    """Returns the value of the `status` field from a JSON string.



    Args:

        json_str: A string representing a JSON object.

    """

    json_obj = json.loads(json_str)

    return json_obj['status']

