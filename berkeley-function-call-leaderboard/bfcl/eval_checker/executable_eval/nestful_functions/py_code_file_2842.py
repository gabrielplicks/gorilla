import json

from typing import Tuple



def extract_file_name_and_extension(file_json: str) -> Tuple[str, str]:

    """Extracts the file name and extension from a JSON string representing a file.



    Args:

        file_json: A JSON string representing a file.



    Returns:

        A tuple containing the file name and extension.

    """

    file_dict = json.loads(file_json)

    name = file_dict['name']

    extension = file_dict['extension']

    return name, extension

