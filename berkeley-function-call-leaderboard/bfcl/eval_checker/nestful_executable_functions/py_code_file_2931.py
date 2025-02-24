import json



def parse_resource_file(filename: str) -> dict:

    """Parses a JSON file and returns the resulting dictionary.



    Args:

        filename: The name of the file to parse.



    Returns:

        The dictionary representing the JSON object contained in the file.

    """

    with open(filename, 'r', encoding='utf-8') as f:

        data = f.read()

        return json.loads(data)

