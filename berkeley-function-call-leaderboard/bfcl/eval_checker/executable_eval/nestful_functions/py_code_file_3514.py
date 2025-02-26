import json



def read_json_array(filename: str) -> list:

    """Reads a file containing a JSON array of objects and returns a list of Python dictionaries that represent each object.



    Args:

        filename: The name of the file to read.

    """

    with open(filename, 'r') as file:

        json_str = file.read()

    return json.loads(json_str)

