import json

from typing import Dict



def create_json_payload(input_dict: Dict[str, int]) -> str:

    """Creates a JSON object from a given input dictionary by iterating through the dictionary's key-value pairs in sorted order and converting the values to strings.



    Args:

        input_dict: The input dictionary.



    Returns:

        A JSON object with keys that are the sorted list of dictionary keys, and values that are the stringified dictionary values.

    """

    sorted_keys = sorted(input_dict.keys())

    stringified_values = [str(value) for value in input_dict.values()]

    json_payload = dict(zip(sorted_keys, stringified_values))

    return json.dumps(json_payload)

