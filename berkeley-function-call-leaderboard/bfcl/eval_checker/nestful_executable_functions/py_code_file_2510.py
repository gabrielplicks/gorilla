import json

from typing import Optional



def get_root_key(file_path: str) -> Optional[str]:

    """

    Returns the root key of a JSON object from a file.

    If the JSON object has multiple root keys, returns the first one.

    If the file path does not exist, returns None.



    Args:

        file_path: The path to the JSON file.

    """

    try:

        with open(file_path, "r") as f:

            json_data = json.load(f)

        root_key = list(json_data.keys())[0]

        return root_key

    except FileNotFoundError:

        return None

