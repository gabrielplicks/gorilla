import os

import json

from typing import Any



def create_files(json_file_path: str) -> None:

    """Creates text files from a JSON file.



    The JSON file contains an array of objects with a 'name' and 'description' field.

    For each object, a new text file is created with the name of the object as the file name

    and the description as the file contents.



    Args:

        json_file_path: The path to the JSON file.

    """

    with open(json_file_path, 'r') as f:

        data: list[dict[str, Any]] = json.load(f)



    for obj in data:

        name: str = obj['name']

        description: str = obj['description']

        file_name: str = name.replace(' ', '_').lower() + '.txt'



        with open(file_name, 'w') as f:

            f.write(description.replace('\\n', '\n'))

