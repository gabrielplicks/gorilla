import json



def make_json_string(dictionary: dict) -> str:

    """Converts a dictionary to a JSON string.



    Args:

        dictionary: The dictionary to convert to JSON.



    Returns:

        A string representing the same data in JSON format.

    """

    return json.dumps(dictionary)

