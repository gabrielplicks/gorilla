import json

from typing import List, Dict



def get_maximum_score(json_file_path: str) -> float:

    """Calculates the maximum score from a JSON file representing a game's scores.

    Args:

        json_file_path: The path to the JSON file.

    """

    with open(json_file_path, 'r') as f:

        data: List[Dict[str, float]] = json.load(f)



    max_score: float = -float('inf')

    for player in data:

        max_score = max(max_score, player['score'])



    return max_score

