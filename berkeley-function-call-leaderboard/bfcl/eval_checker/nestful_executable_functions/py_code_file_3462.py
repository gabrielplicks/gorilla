import json

from typing import Dict



def count_keywords(json_file: str) -> Dict[str, int]:

    """Reads a JSON file and returns a dictionary where the keys are the keywords and the values are the associated counts.



    Args:

        json_file: The path to the JSON file.



    Returns:

        A dictionary where the keys are the keywords and the values are the associated counts.

    """

    with open(json_file, 'r') as f:

        data = json.load(f)

    keywords = data['keywords']

    counts = data['counts']

    return dict(zip(keywords, counts))

