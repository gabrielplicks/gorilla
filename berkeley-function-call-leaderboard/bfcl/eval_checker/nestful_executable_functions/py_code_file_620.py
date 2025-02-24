import json



def deserialize_json(json_file_path: str) -> dict:

    """Reads and deserializes a JSON file into a Python data structure.



    Args:

        json_file_path: The path to the JSON file.



    """

    with open(json_file_path, 'r') as f:

        data = json.load(f)

    return data

