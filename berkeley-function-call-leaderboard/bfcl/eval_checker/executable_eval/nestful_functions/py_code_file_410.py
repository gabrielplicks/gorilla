import json

from typing import Dict, List



def read_and_modify_json(json_file: str) -> List[Dict[str, str]]:

    """Reads and modifies a JSON file.



    Args:

        json_file: The name of the JSON file.



    Returns:

        A list of `frame` objects. The `frame` object is a dictionary with keys

        `id`, `timestamp`, and `data`. The `data` value is a dictionary with keys

        `x` and `y`. The function modifies the `timestamp` value of each `frame`

        object by adding 100 to it.

    """

    with open(json_file, 'r') as file:

        data = json.load(file)

        for frame in data['frames']:

            frame['timestamp'] += 100

            frame['data']['x'] += 100

            frame['data']['y'] += 100

        return data['frames']

