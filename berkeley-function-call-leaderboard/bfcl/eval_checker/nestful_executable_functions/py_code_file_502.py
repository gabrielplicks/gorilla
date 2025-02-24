import json

from typing import List, Tuple, Dict



def generate_tuples_from_json(json_object: str) -> List[Tuple[str, str, int]]:

    """Generates a list of tuples (name, language, score) from the given JSON object whose keys are string representations of integer scores.



    Args:

        json_object: A JSON object as a string.



    Returns:

        A list of tuples (name, language, score).

    """

    data = json.loads(json_object)

    result = []

    for key, value in data.items():

        result.append((key, value['language'], int(value['score'])))

    return result

