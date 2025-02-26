import json

from typing import Dict



def convert_keys_to_int(json_file: str) -> Dict[int, str]:

    """Converts the keys of a dictionary in a JSON file to integers.

    Args:

        json_file: The name of the JSON file.

    """

    with open(json_file, 'r') as f:

        data = json.load(f)

    for key in data.keys():

        data[int(key)] = data.pop(key)

    return data

