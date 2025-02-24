import json

from typing import List, Dict



def json_to_objects(json_file_path: str) -> List[Dict[str, str]]:

    """

    Converts a JSON file to a list of objects with a single key-value pair for each line.

    The key name and its value are the same for each object.



    Args:

        json_file_path: The path to the JSON file.



    Returns:

        A list of objects with a single key-value pair for each line of the input JSON file.

    """

    with open(json_file_path, 'r') as f:

        objects = []

        for line in f:

            data = json.loads(line)

            key = list(data.keys())[0]

            value = data[key]

            objects.append({key: value})

    return objects

