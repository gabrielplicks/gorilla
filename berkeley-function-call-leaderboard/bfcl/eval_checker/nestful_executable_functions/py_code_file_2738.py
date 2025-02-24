from typing import Dict, List



def get_top_scores(data: List[Dict[str, Dict[str, int]]]) -> Dict[str, int]:

    """Returns a dictionary of the top scores across all games.



    The keys are the names of the games and the values are the corresponding top score.



    Args:

        data: A list of dictionaries, with each dictionary having a key called `scores`

            and its value being another dictionary containing game scores.

    """

    top_scores = {}

    for entry in data:

        scores = entry['scores']

        for game, score in scores.items():

            if game not in top_scores or score > top_scores[game]:

                top_scores[game] = score

    return top_scores

