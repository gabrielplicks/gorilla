import json

from typing import Dict, Union



def load_file(file_path: str) -> Union[Dict, None]:

    """Loads a file and returns its content as a dictionary.



    Args:

        file_path: The path to the file.



    Returns:

        A dictionary containing the file content, or None if the file is not found or contains invalid JSON data.

    """

    try:

        with open(file_path, 'r') as file:

            file_content = file.read()

            return json.loads(file_content)

    except FileNotFoundError:

        print(f"File not found: {file_path}")

        return None

    except json.JSONDecodeError:

        print(f"Invalid JSON data in file: {file_path}")

        return None

