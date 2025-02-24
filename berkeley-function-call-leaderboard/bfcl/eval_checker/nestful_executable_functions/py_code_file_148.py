from typing import Dict



def update_score(scores: Dict[str, int], name: str, score: int) -> None:

    """Updates the score for a player in a dictionary.

    If the player is not already in the dictionary, they are added with a default value of 0.

    Args:

        scores: The dictionary containing the scores.

        name: The name of the player.

        score: The new score for the player.

    """

    scores.setdefault(name, 0)

    scores[name] += score

