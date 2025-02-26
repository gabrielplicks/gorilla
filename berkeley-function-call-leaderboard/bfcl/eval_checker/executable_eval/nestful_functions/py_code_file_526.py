import json



def read_objects(filename: str) -> list:

    """Reads a file line by line and returns a list of objects.

    Each line is a JSON string representing an object.

    If any line is invalid, it is skipped.

    Args:

        filename: The name of the file to read.

    """

    objects = []

    with open(filename, 'r') as f:

        for line in f:

            try:

                obj = json.loads(line)

                objects.append(obj)

            except json.JSONDecodeError:

                pass

    return objects

