from typing import List



def create_score_dict(names: List[str], scores: List[float]) -> dict:

    """Creates a dictionary where the keys are taken from the `names` list and the values are taken from the `scores` list.



    Args:

        names: A list of names.

        scores: A list of scores.



    Returns:

        A dictionary where the keys are names and the values are scores.

    """

    result = {name: score for name, score in zip(names, scores)}

    return result

