import json

from typing import List



def read_json_objects(file_path: str) -> List[dict]:

    """Reads JSON objects from a file and returns them as a list.



    Args:

        file_path: The path to the file containing JSON objects.



    Returns:

        A list of JSON objects read from the file.



    Raises:

        FileNotFoundError: If the file does not exist.

        IOError: If there is an error reading the file.

        json.JSONDecodeError: If there is an error decoding the JSON objects.

    """

    try:

        with open(file_path, 'r') as file:

            json_objects = [

                json.loads(line)

                for line in file.read().splitlines()

            ]

        return json_objects

    except (FileNotFoundError, IOError, json.JSONDecodeError) as e:

        print(f"Error reading JSON objects from file: {e}")

        return []

