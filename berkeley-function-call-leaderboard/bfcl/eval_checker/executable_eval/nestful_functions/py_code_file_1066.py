import json



def extract_data_from_event(event: dict) -> dict:

    """

    Extracts data from the 'data' key of an event dict.

    Args:

        event: The event dict containing the 'data' key.

    Returns:

        The data in a Python dict. If the 'data' key is missing or the value is not a valid JSON string,

        an empty dict is returned instead.

    """

    try:

        data = json.loads(event['data'])

        return data

    except (KeyError, ValueError):

        return {}

