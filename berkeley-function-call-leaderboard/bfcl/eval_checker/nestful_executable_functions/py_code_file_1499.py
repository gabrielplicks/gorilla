import json



def pretty_print_json(dictionary: dict) -> str:

    """Converts a Python dictionary to a pretty-printed string representation in JSON format.

    Args:

        dictionary: The dictionary to convert.

    Returns:

        The pretty-printed string representation of the dictionary in JSON format.

    """

    return json.dumps(dictionary, indent=3)

