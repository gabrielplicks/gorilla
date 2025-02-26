import json

from typing import Dict



def read_json_booleans(file_path: str) -> Dict[str, bool]:

    """Reads a JSON file and returns a dictionary where the key is a string and the value is a boolean.



    Args:

        file_path: The path to the JSON file.



    Returns:

        A dictionary where the key is a string and the value is a boolean.

    """

    with open(file_path, "r") as file:

        content = file.read()

        data = json.loads(content)



        boolean_dict = {}

        for key, value in data.items():

            if isinstance(value, bool):

                boolean_dict[key] = value



    return boolean_dict

