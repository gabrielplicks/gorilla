import json

import os



def read_file_as_dictionary(file_path: str) -> dict:

    """Reads a file and returns a dictionary of key-value pairs.



    Args:

        file_path: The path to the file to be read.



    Returns:

        A dictionary of key-value pairs.



    Raises:

        Exception: If the file does not exist or is not a regular file.

    """

    try:

        if not os.path.isfile(file_path):

            raise Exception(f"File does not exist or is not a regular file: {file_path}")



        with open(file_path, "r") as f:

            content = f.read()

            return json.loads(content)

    except Exception as e:

        print(f"Error reading file: {e}")

        return {}

