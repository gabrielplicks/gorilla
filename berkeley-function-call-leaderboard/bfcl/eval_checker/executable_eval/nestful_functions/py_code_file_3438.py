import json

from typing import List, Dict



def load_json_file(path: str) -> List[Dict]:

    """Loads a JSON file and returns a list of dictionaries, each containing the data from a JSON object in the file.



    Args:

        path: The file path of the JSON file.



    Returns:

        A list of dictionaries, where each dictionary contains the data from a JSON object in the file.

    """

    results = []

    with open(path, 'r') as f:

        for line in f:

            data = json.loads(line)

            results.append(data)

    return results

