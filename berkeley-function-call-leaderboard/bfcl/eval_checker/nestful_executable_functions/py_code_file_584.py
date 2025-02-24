import json



def serialize_dictionary(dictionary: dict) -> str:

    """

    Serializes a dictionary into JSON format, with each key-value pair on a separate line and indented by 4 spaces.

    Args:

        dictionary: The dictionary to be serialized.

    """

    return json.dumps(dictionary, indent=4, separators=(',', ': '))

