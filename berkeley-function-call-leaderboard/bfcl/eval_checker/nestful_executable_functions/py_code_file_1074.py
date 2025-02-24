import json

from typing import Dict, List



def load_json_lines(filename: str) -> List[Dict[str, str]]:

    """Loads a file containing JSON lines into a dictionary, where each line represents an object in JSON format.

    The function returns a list of objects. Each object has two keys: "id" and "data", where "id" is a string and "data" is a dictionary of string keys and string values.

    Args:

        filename: The name of the file to load.

    """

    with open(filename) as f:

        objects = []

        for line in f:

            obj = json.loads(line)

            objects.append({"id": obj["id"], "data": obj["data"]})

        return objects

