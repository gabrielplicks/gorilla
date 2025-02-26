import json

import re

import typing



def load_json_objects(file_path: str) -> typing.List[typing.Any]:

    """Loads JSON objects from a file.



    Args:

        file_path: The path to the file containing the JSON data.



    Returns:

        A list of JSON objects.

    """

    objects = []

    comment_pattern = re.compile(r'^#.*$')



    with open(file_path, 'r') as file:

        for line in file:

            line = line.strip()

            if not comment_pattern.match(line):

                objects.append(json.loads(line))



    return objects

