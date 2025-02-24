import json

import tempfile



def create_temporary_json_file(dictionary: dict) -> tempfile._TemporaryFileWrapper:

    """Creates a temporary json file and writes a dictionary to it.



    Args:

        dictionary: The dictionary to write to the temporary file.



    Returns:

        The temporary file object.

    """

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:

        json.dump(dictionary, temp_file)

    return temp_file

