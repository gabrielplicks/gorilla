import os

import json



def read_json(file_path: str) -> dict:

    """Reads a JSON file and returns the parsed JSON object.



    Args:

        file_path: The path to the JSON file.



    Returns:

        The parsed JSON object if the file exists and the JSON content is valid.

        None if the file does not exist or the JSON content is invalid.

    """

    if not os.path.isfile(file_path):

        return None

    with open(file_path, "r") as file:

        try:

            return json.loads(file.read())

        except json.decoder.JSONDecodeError:

            return None

